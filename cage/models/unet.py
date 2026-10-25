"""U-Net denoiser for 128×128 RGB images (CAGE pipeline)."""

import torch
import torch.nn as nn


class _DoubleConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """Three-level U-Net with channels 16, 32, 64, and 128."""

    def __init__(self, channels: int = 3) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.enc1 = _DoubleConvBlock(channels, 16)
        self.enc2 = _DoubleConvBlock(16, 32)
        self.enc3 = _DoubleConvBlock(32, 64)
        self.bottleneck = _DoubleConvBlock(64, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = _DoubleConvBlock(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = _DoubleConvBlock(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = _DoubleConvBlock(32, 16)
        self.out = nn.Sequential(nn.Conv2d(16, channels, kernel_size=1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)
