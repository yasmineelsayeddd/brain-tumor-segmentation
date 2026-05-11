"""U-Net from scratch for multi-class brain tumor segmentation.

Architecture: Ronneberger et al., 2015 — adapted for:
  - 4 input channels (MRI modalities)
  - 4 output classes (background, NCR/NET, edema, enhancing)
  - Batch normalisation instead of dropout
  - Bilinear upsampling (more stable than transposed conv)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Conv → BN → ReLU → Conv → BN → ReLU."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int | None = None) -> None:
        super().__init__()
        mid = mid_channels or out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """MaxPool2d → DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    """Bilinear upsample → concatenate skip → DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # after concat, channels = in_channels (skip) + in_channels // 2 (upsampled)
        self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad x if skip has a different spatial size (can happen with odd input dims)
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                       diff_h // 2, diff_h - diff_h // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """Standard U-Net with configurable base channels.

    Args:
        in_channels:  Number of input modalities (4 for BraTS).
        out_channels: Number of output classes (4 for BraTS).
        base_channels: Width of the first encoder block. Doubles at each level.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        b = base_channels

        # Encoder
        self.enc1 = DoubleConv(in_channels, b)
        self.enc2 = Down(b, b * 2)
        self.enc3 = Down(b * 2, b * 4)
        self.enc4 = Down(b * 4, b * 8)

        # Bottleneck
        self.bottleneck = Down(b * 8, b * 16)

        # Decoder
        self.dec4 = Up(b * 16, b * 8)
        self.dec3 = Up(b * 8,  b * 4)
        self.dec2 = Up(b * 4,  b * 2)
        self.dec1 = Up(b * 2,  b)

        # Output projection
        self.head = nn.Conv2d(b, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        # Bottleneck
        b = self.bottleneck(s4)

        # Decoder with skip connections
        x = self.dec4(b,  s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        return self.head(x)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
