import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange

__all__ = [
    "WithBias_LayerNorm_ODAM",
    "LayerNorm_ODAM",
    "MultiScaleDWConv_ODAM",
    "OrthogonalDynamicAttention",
    "ODAM",
]


# ---------------------------
# Utils
# ---------------------------
def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


# ---------------------------
# LayerNorm
# ---------------------------
class WithBias_LayerNorm_ODAM(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm_ODAM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = WithBias_LayerNorm_ODAM(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# ---------------------------
# Multi-Scale Depthwise Conv
# ---------------------------
class MultiScaleDWConv_ODAM(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.convs = nn.ModuleList([
            nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim),
            nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim),
            nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim),
            nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim),
            nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim),
            nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim),
        ])

        self.weight = nn.Parameter(torch.ones(len(self.convs)))

    def forward(self, x):
        weights = F.softmax(self.weight, dim=0)

        out = 0
        for i, conv in enumerate(self.convs):
            out = out + weights[i] * conv(x)

        return out


# ---------------------------
# ODAM Core
# ---------------------------
class OrthogonalDynamicAttention(nn.Module):

    def __init__(self, dim, num_heads=8):
        super().__init__()

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.tau = nn.Parameter(torch.log(torch.tensor(10.0)))

        self.norm1 = LayerNorm_ODAM(dim)
        self.norm2 = LayerNorm_ODAM(dim)

        self.ms1 = MultiScaleDWConv_ODAM(dim)
        self.ms2 = MultiScaleDWConv_ODAM(dim)

        self.q_proj = nn.Conv2d(dim, dim, 1)
        self.k_proj = nn.Conv2d(dim, dim, 1)
        self.v_proj = nn.Conv2d(dim, dim, 1)

        self.project_out = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x1 = self.norm1(x)
        x2 = self.norm2(x)

        f1 = self.ms1(x1)
        f2 = self.ms2(x2)

        q1 = self.q_proj(f2)
        k1 = self.k_proj(f1)
        v1 = self.v_proj(f1)

        q2 = self.q_proj(f1)
        k2 = self.k_proj(f2)
        v2 = self.v_proj(f2)

        q1 = rearrange(q1, "b (h c) H W -> b h H (W c)", h=self.num_heads)
        k1 = rearrange(k1, "b (h c) H W -> b h H (W c)", h=self.num_heads)
        v1 = rearrange(v1, "b (h c) H W -> b h H (W c)", h=self.num_heads)

        q2 = rearrange(q2, "b (h c) H W -> b h W (H c)", h=self.num_heads)
        k2 = rearrange(k2, "b (h c) H W -> b h W (H c)", h=self.num_heads)
        v2 = rearrange(v2, "b (h c) H W -> b h W (H c)", h=self.num_heads)

        q1 = F.normalize(q1, dim=-1)
        k1 = F.normalize(k1, dim=-1)
        q2 = F.normalize(q2, dim=-1)
        k2 = F.normalize(k2, dim=-1)

        tau = self.tau.exp()

        attn1 = (q1 @ k1.transpose(-2, -1)) * tau
        attn1 = attn1.softmax(dim=-1)
        out1 = attn1 @ v1

        attn2 = (q2 @ k2.transpose(-2, -1)) * tau
        attn2 = attn2.softmax(dim=-1)
        out2 = attn2 @ v2

        out1 = rearrange(
            out1,
            "b h H (W c) -> b (h c) H W",
            h=self.num_heads,
            H=h,
            W=w,
        )
        out2 = rearrange(
            out2,
            "b h W (H c) -> b (h c) H W",
            h=self.num_heads,
            H=h,
            W=w,
        )

        out = self.project_out(out1 + out2)

        return out + x


# ---------------------------
# YAML Top-Level Module
# ---------------------------
class ODAM(nn.Module):
    """ODAM: Orthogonal Dynamic Attention Module"""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0, act=nn.GELU(), normalize_before=False):
        super().__init__()

        if cm is None or cm > 4 * c1:
            cm = 4 * c1

        self.odam = OrthogonalDynamicAttention(dim=c1, num_heads=num_heads)

        self.norm2 = LayerNorm_ODAM(c1)
        self.fc1 = nn.Conv2d(c1, cm, 1)
        self.fc2 = nn.Conv2d(cm, c1, 1)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = act

    def forward(self, x, *args, **kwargs):
        x = self.odam(x)

        x_ffn = self.norm2(x)
        x_ffn = self.fc2(self.dropout1(self.act(self.fc1(x_ffn))))
        x = x + self.dropout2(x_ffn)

        return x
