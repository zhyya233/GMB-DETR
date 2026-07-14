import pywt
import torch
from torch import nn
import torch.nn.functional as F

__all__ = [
    'HGWTConv2d',
    'HG_WTConv',
    'HG_ResBlock',
    'HG_ResLayer',
    'HG-WTConv',
    'HG-ResLayer',
]


# ==========================================

# ==========================================
def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ==========================================

# ==========================================
def create_wavelet_filter(wave, channels, dtype=torch.float32):
    w = pywt.Wavelet(wave)

    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=dtype)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=dtype)
    LL = torch.outer(dec_lo, dec_lo)
    LH = torch.outer(dec_lo, dec_hi)
    HL = torch.outer(dec_hi, dec_lo)
    HH = torch.outer(dec_hi, dec_hi)
    k_dec = torch.stack([LL, LH, HL, HH], dim=0)[:, None, :, :].repeat(channels, 1, 1, 1)

    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=dtype).flip(0)
    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=dtype).flip(0)
    iLL = torch.outer(rec_lo, rec_lo)
    iLH = torch.outer(rec_lo, rec_hi)
    iHL = torch.outer(rec_hi, rec_lo)
    iHH = torch.outer(rec_hi, rec_hi)
    k_rec = torch.stack([iLL, iLH, iHL, iHH], dim=0)[:, None, :, :].repeat(channels, 1, 1, 1)

    return k_dec, k_rec


class EfficientBandGate(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, hidden, 1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv1d(hidden, channels, 1, bias=True)
        )
        self.band_bias = nn.Parameter(torch.zeros(1, channels, 4, 1, 1))

    def forward(self, x):
        b, c, n, _, _ = x.shape
        s = x.mean(dim=(3, 4))
        g = self.fc(s).view(b, c, n, 1, 1)
        g = torch.sigmoid(g + self.band_bias)
        return x * g


class LiteLevelFusion(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.f = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * channels, channels, 1, groups=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, a, b):
        alpha = self.f(torch.cat([a, b], dim=1))
        return alpha * a + (1.0 - alpha) * b


class HGWTConv2d(nn.Module):

    def __init__(self, channels, kernel_size=5, stride=1, wt_levels=1, wt_type='db1', band_reduction=16):
        super().__init__()
        self.c = channels
        self.wt_levels = wt_levels
        self.stride = stride

        wt_filter, iwt_filter = create_wavelet_filter(wt_type, self.c, torch.float32)
        self.register_buffer("wt_filter", wt_filter)
        self.register_buffer("iwt_filter", iwt_filter)

        p = kernel_size // 2
        self.base_conv = nn.Conv2d(self.c, self.c, kernel_size, stride=1, padding=p, groups=self.c, bias=False)
        self.base_scale = nn.Parameter(torch.ones(1, self.c, 1, 1))

        self.wavelet_convs = nn.ModuleList([
            nn.Conv2d(self.c * 4, self.c * 4, kernel_size, stride=1, padding=p, groups=self.c * 4, bias=False)
            for _ in range(wt_levels)
        ])
        self.wavelet_scale = nn.ParameterList([
            nn.Parameter(torch.ones(1, self.c * 4, 1, 1) * 0.1)
            for _ in range(wt_levels)
        ])

        self.band_gates = nn.ModuleList([
            EfficientBandGate(self.c, reduction=band_reduction)
            for _ in range(wt_levels)
        ])
        self.level_fusions = nn.ModuleList([
            LiteLevelFusion(self.c)
            for _ in range(max(0, wt_levels - 1))
        ])

    def wavelet_transform(self, x):
        k = self.wt_filter.shape[-1]
        pad = k // 2 - 1
        y = F.conv2d(x, self.wt_filter, stride=2, padding=pad, groups=self.c)
        return y.view(y.shape[0], self.c, 4, y.shape[2], y.shape[3])

    def inverse_wavelet_transform(self, x):
        b, c, _, h, w = x.shape
        k = self.iwt_filter.shape[-1]
        pad = k // 2 - 1
        y = x.view(b, c * 4, h, w)
        return F.conv_transpose2d(y, self.iwt_filter, stride=2, padding=pad, groups=self.c)

    def forward(self, x):
        ll_stack, h_stack, shape_stack = [], [], []
        curr = x

        for i in range(self.wt_levels):
            shape_stack.append(curr.shape)
            pad_h, pad_w = curr.shape[-2] % 2, curr.shape[-1] % 2
            curr = F.pad(curr, (0, pad_w, 0, pad_h), mode='reflect')

            bands = self.wavelet_transform(curr)
            curr = bands[:, :, 0]

            t = bands.reshape(bands.shape[0], self.c * 4, bands.shape[3], bands.shape[4])
            t = self.wavelet_convs[i](t) * self.wavelet_scale[i]
            t = t.view_as(bands)

            t = self.band_gates[i](t)

            ll_stack.append(t[:, :, 0])
            h_stack.append(t[:, :, 1:4])

        next_ll = None
        for i in range(self.wt_levels - 1, -1, -1):
            ll_i = ll_stack.pop()
            h_i = h_stack.pop()
            orig = shape_stack.pop()

            if next_ll is None:
                fused_ll = ll_i
            else:
                fused_ll = self.level_fusions[i](ll_i, next_ll)

            cat = torch.cat([fused_ll.unsqueeze(2), h_i], dim=2)
            next_ll = self.inverse_wavelet_transform(cat)
            next_ll = next_ll[:, :, :orig[2], :orig[3]]

        out = self.base_conv(x) * self.base_scale + next_ll

        if self.stride > 1:
            out = out[:, :, ::self.stride, ::self.stride]

        return out


# ==========================================

# ==========================================
class HG_WTConv(nn.Module):
    """HG-WTConv: Hierarchical Gated Wavelet Convolution"""

    def __init__(self, c1, c2, k=3, s=1, wt_levels=1):
        super().__init__()
        self.wt = HGWTConv2d(channels=c1, kernel_size=k, stride=s, wt_levels=wt_levels)
        self.pw = Conv(c1, c2, k=1, s=1)

    def forward(self, x):
        return self.pw(self.wt(x))


class HG_ResBlock(nn.Module):
    """HG-ResBlock: Hierarchical Gated Residual Block"""

    def __init__(self, c1, c2, s=1, e=4):
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1)
        self.cv3 = HG_WTConv(c2, c3, k=3, s=1)
        self.shortcut = nn.Sequential(
            Conv(c1, c3, k=1, s=s, act=False)
        ) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class HG_ResLayer(nn.Module):
    """HG-ResLayer: Hierarchical Gated Residual Layer"""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [HG_ResBlock(c1, c2, s, e=e)]
            blocks.extend([
                HG_ResBlock(e * c2, c2, 1, e=e)
                for _ in range(n - 1)
            ])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        return self.layer(x)

globals()['HG-WTConv'] = HG_WTConv
globals()['HG-ResLayer'] = HG_ResLayer
