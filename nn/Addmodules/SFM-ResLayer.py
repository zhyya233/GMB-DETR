import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ultralytics.nn.modules.conv import Conv

__all__ = [
    "StarReLU",
    "KernelSpatialModulation_Global",
    "FDConv",
    "SFM_Block",
    "SFM_ResBlock",
    "SFM_ResLayer",
    "SFM-Block",
    "SFM-ResLayer",
]


class StarReLU(nn.Module):

    def __init__(self, scale_value=1.0, bias_value=0.0):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.scale = nn.Parameter(scale_value * torch.ones(1))
        self.bias = nn.Parameter(bias_value * torch.ones(1))

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias


def get_fft2freq(d1, d2, use_rfft=False):
    freq_h = torch.fft.fftfreq(d1)
    freq_w = torch.fft.rfftfreq(d2) if use_rfft else torch.fft.fftfreq(d2)
    freq_hw = torch.stack(torch.meshgrid(freq_h, freq_w, indexing="ij"), dim=-1)
    dist = torch.norm(freq_hw, dim=-1)
    sorted_dist, indices = torch.sort(dist.view(-1))

    if use_rfft:
        d2 = d2 // 2 + 1

    sorted_coords = torch.stack([indices // d2, indices % d2], dim=-1)
    return sorted_coords.permute(1, 0), freq_hw


class KernelSpatialModulation_Global(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, kernel_num=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        attention_channel = max(int(in_planes * 0.0625), 16)

        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = StarReLU()

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1)
        self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1)
        self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1)
        self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.normal_(self.channel_fc.weight, std=1e-6)
        nn.init.normal_(self.filter_fc.weight, std=1e-6)
        nn.init.normal_(self.spatial_fc.weight, std=1e-6)
        nn.init.normal_(self.kernel_fc.weight, std=1e-6)

    def forward(self, x):
        avg_x = F.adaptive_avg_pool2d(x, 1)
        avg_x = self.relu(self.bn(self.fc(avg_x)))

        c_att = torch.sigmoid(
            self.channel_fc(avg_x).view(x.size(0), 1, 1, -1, 1, 1)
        ) * 2.0
        f_att = torch.sigmoid(
            self.filter_fc(avg_x).view(x.size(0), 1, -1, 1, 1, 1)
        ) * 2.0
        s_att = torch.sigmoid(
            self.spatial_fc(avg_x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        ) * 2.0

        k_logit = self.kernel_fc(avg_x).view(x.size(0), -1, 1, 1, 1, 1)
        k_att = F.softmax(k_logit, dim=1)

        return c_att, f_att, s_att, k_att


class FDConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, kernel_num=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kernel_num = kernel_num
        self.param_ratio = 1

        self.KSM_Global = KernelSpatialModulation_Global(
            in_channels, out_channels, kernel_size, kernel_num
        )

        weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        nn.init.kaiming_normal_(weight, mode="fan_out", nonlinearity="relu")

        d1, d2, k1, k2 = out_channels, in_channels, kernel_size, kernel_size
        weight = weight.permute(0, 2, 1, 3).reshape(d1 * k1, d2 * k2)
        weight_rfft = torch.fft.rfft2(weight, dim=(0, 1))

        weight_rfft = torch.stack([weight_rfft.real, weight_rfft.imag], dim=-1)[None,].repeat(
            self.param_ratio, 1, 1, 1
        )
        weight_rfft = weight_rfft / (min(out_channels, in_channels) // 2)
        self.dft_weight = nn.Parameter(weight_rfft, requires_grad=True)

        freq_indices, _ = get_fft2freq(d1 * k1, d2 * k2, use_rfft=True)
        self.register_buffer("freq_indices", freq_indices.reshape(2, self.kernel_num, -1))
        self.alpha = min(out_channels, in_channels) // 2 * self.kernel_num

    def forward(self, x):
        b, c_in, h, w = x.size()

        c_att, f_att, s_att, k_att = self.KSM_Global(x)

        DFT_map = torch.zeros(
            (
                b,
                self.out_channels * self.kernel_size,
                self.in_channels * self.kernel_size // 2 + 1,
                2,
            ),
            device=x.device,
        )

        k_att = k_att.reshape(b, self.param_ratio, self.kernel_num, -1)

        idx = self.freq_indices
        w_dft = self.dft_weight[0][idx[0, :, :], idx[1, :, :]][None] * self.alpha

        res = torch.stack(
            [w_dft[..., 0] * k_att[:, 0], w_dft[..., 1] * k_att[:, 0]],
            dim=-1,
        )
        DFT_map[:, idx[0, :, :], idx[1, :, :]] += res

        adaptive_weights = torch.fft.irfft2(torch.view_as_complex(DFT_map), dim=(1, 2))
        adaptive_weights = adaptive_weights.reshape(
            b,
            1,
            self.out_channels,
            self.kernel_size,
            self.in_channels,
            self.kernel_size,
        ).permute(0, 1, 2, 4, 3, 5)

        aggregate_weight = s_att * c_att * f_att * adaptive_weights
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_channels, self.kernel_size, self.kernel_size]
        )

        x_reshaped = x.reshape(1, -1, h, w)

        out = F.conv2d(
            x_reshaped,
            weight=aggregate_weight,
            stride=self.stride,
            padding=self.padding,
            groups=b,
        )

        return out.view(b, self.out_channels, out.size(-2), out.size(-1))


class SFM_ResBlock(nn.Module):
    """SFM-ResBlock: Spatial-Frequency Modulation Residual Block"""

    def __init__(self, c1, c2, s=1, e=4):
        super().__init__()
        c3 = int(e * c2)

        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)

        self.sfm = FDConv(c1, c1)

        self.shortcut = nn.Sequential(
            Conv(c1, c3, k=1, s=s, act=False)
        ) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        return F.relu(self.cv3(self.cv2(self.cv1(self.sfm(x)))) + self.shortcut(x))


class SFM_ResLayer(nn.Module):
    """SFM-ResLayer: Spatial-Frequency Modulation Residual Layer"""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        else:
            blocks = [SFM_ResBlock(c1, c2, s, e=e)]
            blocks.extend([
                SFM_ResBlock(e * c2, c2, 1, e=e)
                for _ in range(n - 1)
            ])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        return self.layer(x)


class SFM_Block(nn.Module):
    """SFM-Block: Spatial-Frequency Modulation Block"""

    def __init__(self, c1, c2, shortcut=True):
        super().__init__()
        self.cv1 = Conv(c1, c2, k=1, s=1)
        self.sfm = FDConv(c2, c2, kernel_size=3)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.sfm(self.cv1(x))
        return x + y if self.add else y
