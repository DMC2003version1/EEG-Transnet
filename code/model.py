import math
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from einops import rearrange
from utils import numberClassChannel


class SincConv2D(nn.Module):
    """Learnable band-pass filter bank applied independently to each EEG channel."""

    def __init__(self, out_channels=18, kernel_size=31, sample_rate=250,
                 min_hz=4.0, max_hz=40.0, min_band_hz=2.0):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("SincConv2D requires an odd kernel size.")

        self.out_channels = out_channels
        self.sample_rate = float(sample_rate)
        self.min_hz = float(min_hz)
        self.max_hz = float(max_hz)
        self.min_band_hz = float(min_band_hz)

        frequencies = torch.linspace(0.1, 0.9, out_channels)
        self.low_hz = nn.Parameter(torch.logit(frequencies))
        self.band_hz = nn.Parameter(torch.zeros(out_channels))
        time = (torch.arange(kernel_size) - kernel_size // 2) / self.sample_rate
        self.register_buffer("time", time)
        self.register_buffer("window", torch.hamming_window(kernel_size, periodic=False))

    def forward(self, x: Tensor) -> Tensor:
        available = self.max_hz - self.min_hz - self.min_band_hz
        low = self.min_hz + available * torch.sigmoid(self.low_hz)
        high = low + self.min_band_hz + (self.max_hz - low - self.min_band_hz) * torch.sigmoid(self.band_hz)

        time = self.time.unsqueeze(0)
        low_pass_high = 2 * high.unsqueeze(1) * torch.sinc(2 * high.unsqueeze(1) * time)
        low_pass_low = 2 * low.unsqueeze(1) * torch.sinc(2 * low.unsqueeze(1) * time)
        filters = (low_pass_high - low_pass_low) * self.window.unsqueeze(0)
        filters = filters / filters.abs().sum(dim=1, keepdim=True).clamp_min(1e-8)
        return F.conv2d(x, filters[:, None, None, :], padding=(0, self.time.numel() // 2))


class MixedDepthTemporalCNN(nn.Module):
    """A light multi-scale temporal encoder following the adaptive Sinc filter bank."""

    def __init__(self, in_channels, filters_per_branch=8,
                 kernel_sizes=(15, 31, 63, 125)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, filters_per_branch, (1, kernel),
                          padding=(0, kernel // 2), bias=False),
                nn.BatchNorm2d(filters_per_branch),
                nn.ELU(),
            )
            for kernel in kernel_sizes
        ])
        self.out_channels = filters_per_branch * len(kernel_sizes)

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([branch(x) for branch in self.branches], dim=1)


class LogVarianceTokenizer(nn.Module):
    """Converts spatial-filter responses into fixed-length ERD/ERS power tokens."""

    def __init__(self, in_channels, emb_size, num_tokens=8, dropout_rate=0.3):
        super().__init__()
        self.num_tokens = num_tokens
        self.projection = nn.Sequential(
            nn.Linear(in_channels, emb_size),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, feature_channels, 1, time).  E[x^2] - E[x]^2 is variance.
        x = x.squeeze(2)
        mean = F.adaptive_avg_pool1d(x, self.num_tokens)
        mean_square = F.adaptive_avg_pool1d(x.square(), self.num_tokens)
        log_variance = torch.log((mean_square - mean.square()).clamp_min(1e-6))
        return self.projection(log_variance.transpose(1, 2))


class PatchEmbeddingCNN(nn.Module):
    """Frequency-adaptive, multi-scale CNN stem used before the unchanged SATrans encoder."""

    def __init__(self, f1=8, kernel_size=64, D=2, pooling_size1=8, pooling_size2=8,
                 dropout_rate=0.3, number_channel=22, emb_size=40, sample_rate=250,
                 sinc_filters=18, filters_per_branch=8, num_tokens=8):
        super().__init__()
        del f1, kernel_size, pooling_size1, pooling_size2  # retained for backward-compatible callers
        self.sinc = SincConv2D(sinc_filters, kernel_size=31, sample_rate=sample_rate)
        self.temporal = MixedDepthTemporalCNN(sinc_filters, filters_per_branch)
        temporal_channels = self.temporal.out_channels
        spatial_channels = temporal_channels * D
        self.spatial = nn.Sequential(
            nn.Conv2d(temporal_channels, spatial_channels, (number_channel, 1),
                      groups=temporal_channels, bias=False),
            nn.BatchNorm2d(spatial_channels),
            nn.ELU(),
            nn.Dropout(dropout_rate),
        )
        self.tokenizer = LogVarianceTokenizer(spatial_channels, emb_size, num_tokens, dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        x = self.sinc(x)
        x = self.temporal(x)
        x = self.spatial(x)
        return self.tokenizer(x)
    




class Attention(nn.Module):
    def __init__(self, dim, num_heads, topk_ratios=[0.5, 2/3, 3/4, 4/5], dropout=0.3, kernel_size=3, norm_type='layer'):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.topk_ratios = topk_ratios
        self.dropout = dropout

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv1d(dim, dim * 3, kernel_size=1)
        self.qkv_dwconv = nn.Conv1d(dim * 3, dim * 3, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=dim * 3)
        self.project_out = nn.Conv1d(dim, dim, kernel_size=1)
        self.attn_drop = nn.Dropout(dropout)


        self.attn_weights = nn.Parameter(torch.ones(len(topk_ratios)) / len(topk_ratios))

        self.norm_type = norm_type
        if norm_type == 'layer':
            self.norm = None  
        elif norm_type == 'batch':
            self.norm = nn.BatchNorm1d(dim)
        else:
            self.norm = lambda x: F.normalize(x, dim=-1)

    def forward(self, x):
        x = x.permute((0, 2, 1))
        b, c, h = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h -> b head c h', head=self.num_heads)
        k = rearrange(k, 'b (head c) h -> b head c h', head=self.num_heads)
        v = rearrange(v, 'b (head c) h -> b head c h', head=self.num_heads)


        if self.norm_type == 'layer' and self.norm is None:
            normalized_shape = q.shape[-1:]  
            self.norm = nn.LayerNorm(normalized_shape).to(q.device)  

        q = self.norm(q)
        k = self.norm(k)

        _, _, C, _ = q.shape

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        topk_indices = [
            torch.topk(attn, k=int(C * ratio), dim=-1, largest=True)[1]
            for ratio in self.topk_ratios
        ]

        masks = [
            torch.zeros_like(attn, device=x.device).scatter_(-1, index, 1.)
            for index in topk_indices
        ]


        attns = [
            torch.where(mask > 0, attn, torch.full_like(attn, float('-inf'))).softmax(dim=-1)
            for mask in masks
        ]


        if self.training:
            attns = [self.attn_drop(attn) for attn in attns]


        outs = [attn @ v for attn in attns]
        out = sum(out * weight for out, weight in zip(outs, self.attn_weights))

        out = rearrange(out, 'b head c h -> b (head c) h', head=self.num_heads, h=h)
        out = self.project_out(out)
        out = self.attn_drop(out) 
        out = out.permute((0, 2, 1))
        return out

# PointWise FFN
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


#Classification
class ClassificationHead(nn.Sequential):
    def __init__(self, flatten_number, n_classes):
        super().__init__()
        self.fc = nn.Sequential(
            # nn.Linear(flatten_number, 256),
            nn.Dropout(0.5),
            nn.Linear(flatten_number, n_classes),
           # nn.Softmax(dim=1),
        )

    def forward(self, x):
        out = self.fc(x)
        
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn, emb_size, drop_p):
        super().__init__()
        self.fn = fn
        self.drop = nn.Dropout(drop_p)
        self.layernorm = nn.LayerNorm(emb_size)

    def forward(self, x, **kwargs):
        x_input = x
        res = self.fn(x, **kwargs)
        
        out = self.layernorm(self.drop(res)+x_input)
        return out

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=4,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                Attention(emb_size, num_heads), 
                ), emb_size, drop_p),
     
            ResidualAdd(nn.Sequential(
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                ), emb_size, drop_p)
            
            )    
        
        
class TransformerEncoder(nn.Sequential):
    def __init__(self, heads, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size, heads) for _ in range(depth)])




class BranchEEGNetTransformer(nn.Sequential):
    def __init__(self, heads=4, 
                 depth=6, 
                 emb_size=40, 
                 number_channel=22,
                 f1 = 20,
                 kernel_size = 64,
                 D = 2,
                 pooling_size1 = 8,
                 pooling_size2 = 8,
                 dropout_rate = 0.3,
                 num_tokens=8,
                 **kwargs):
        super().__init__(
            PatchEmbeddingCNN(f1=f1, 
                                 kernel_size=kernel_size,
                                 D=D, 
                                 pooling_size1=pooling_size1, 
                                 pooling_size2=pooling_size2,
                                 dropout_rate=dropout_rate,
                                 number_channel=number_channel,
                                 emb_size=emb_size,
                                 num_tokens=num_tokens),
#             TransformerEncoder(heads, depth, emb_size),
        )



    

        
class PositioinalEncoding(nn.Module):
    def __init__(self, embedding, length=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.encoding = nn.Parameter(torch.randn(1, length, embedding))
    def forward(self, x): # x-> [batch, embedding, length]
        x = x + self.encoding[:, :x.shape[1], :].to(x.device)
        return self.dropout(x)        
        
   
    
class EEGTransformer(nn.Module):
    def __init__(self, heads=4, 
                 emb_size=40,
                 depth=6, 
                 database_type='A', 
                 eeg1_f1=20,
                 eeg1_kernel_size=64,
                 eeg1_D=2,
                 eeg1_pooling_size1=8,
                 eeg1_pooling_size2=8,
                 eeg1_dropout_rate=0.3,
                 eeg1_number_channel=22,
                 flatten_eeg1=None,
                 num_tokens=8,
                 **kwargs):
        super().__init__()
        self.number_class, self.number_channel = numberClassChannel(database_type)
        self.emb_size = emb_size
        self.flatten_eeg1 = flatten_eeg1 or emb_size * num_tokens
        self.flatten = nn.Flatten()
        
        self.cnn = BranchEEGNetTransformer(heads, depth, emb_size, number_channel=self.number_channel,
                                              f1=eeg1_f1,
                                              kernel_size=eeg1_kernel_size,
                                              D=eeg1_D,
                                              pooling_size1=eeg1_pooling_size1,
                                              pooling_size2=eeg1_pooling_size2,
                                              dropout_rate=eeg1_dropout_rate,
                                              num_tokens=num_tokens)
        

        self.position = PositioinalEncoding(emb_size, 100, dropout=0.1)
        self.trans = TransformerEncoder(heads, depth, emb_size)


        
         
        self.flatten = nn.Flatten()
        self.classification = ClassificationHead(self.flatten_eeg1, self.number_class) # FLATTEN_EEGNet + FLATTEN_cnn_module

    def forward(self, x):


        cnn = self.cnn(x)
        # add label 
        cnn = cnn * math.sqrt(self.emb_size)

        features = self.position(cnn)
        features = self.trans(features)

        features = cnn + features

        # features = self.cnn_output(features)
        out = self.classification(self.flatten(features))

        return features, out
    
