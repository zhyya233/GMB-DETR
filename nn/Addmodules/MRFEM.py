import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv

__all__ = [
    "MRFEMBlock",
    "MRFEM",
]


class MRFEMBlock(nn.Module):
    """
    Multi-Receptive-Field Enhancement Block.

    1. Decouples BatchNorm statistics across receptive-field branches.
    2. Uses a single residual connection to reduce feature degradation.
    3. Adaptively fuses multi-scale receptive fields with learnable scalar weights.
    """

    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        c_ = int(c2 * e)

        self.w1 = nn.Parameter(torch.empty(c_, c1, 1, 1))
        self.w2 = nn.Parameter(torch.empty(c2, c_, 3, 3))
        nn.init.kaiming_uniform_(self.w1, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.w2, nonlinearity="relu")

        self.bn1 = nn.BatchNorm2d(c_)

        self.bn2_1 = nn.BatchNorm2d(c2)
        self.bn2_2 = nn.BatchNorm2d(c2)
        self.bn2_3 = nn.BatchNorm2d(c2)

        self.alpha = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.act = nn.SiLU()

    def forward(self, x):
        out = F.conv2d(x, self.w1)
        out = self.act(self.bn1(out))

        b1 = self.bn2_1(F.conv2d(out, self.w2, padding=1, dilation=1))
        b2 = self.bn2_2(F.conv2d(out, self.w2, padding=2, dilation=2))
        b3 = self.bn2_3(F.conv2d(out, self.w2, padding=3, dilation=3))

        weights = self.alpha
        fused = self.act(weights[0] * b1 + weights[1] * b2 + weights[2] * b3)

        return fused + x


class MRFEM(nn.Module):
    """Multi-Receptive-Field Enhancement Module."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)

        self.m = nn.Sequential(
            *(MRFEMBlock(c_, c_, e=1.0) for _ in range(n))
        )

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
