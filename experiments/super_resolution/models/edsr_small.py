"""
EDSR-Small: Enhanced Deep Residual Networks for Single Image Super-Resolution
=============================================================================
Reference: Lim et al., "Enhanced Deep Residual Networks for Single Image
Super-Resolution", CVPRW 2017.
"""

import math
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Residual block without Batch Normalization, with residual scaling."""
    def __init__(self, channels: int, res_scale: float = 0.1):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.conv2(self.relu(self.conv1(x)))
        return x + res * self.res_scale


class Upsampler(nn.Sequential):
    """PixelShuffle upsampler module supporting 2x and 4x scaling."""
    def __init__(self, scale: int, channels: int):
        layers = []
        if (scale & (scale - 1)) == 0:  # Power of 2 (2x, 4x, 8x)
            for _ in range(int(math.log2(scale))):
                layers.append(nn.Conv2d(channels, 4 * channels, kernel_size=3, padding=1))
                layers.append(nn.PixelShuffle(2))
                layers.append(nn.PReLU(channels))
        elif scale == 3:
            layers.append(nn.Conv2d(channels, 9 * channels, kernel_size=3, padding=1))
            layers.append(nn.PixelShuffle(3))
            layers.append(nn.PReLU(channels))
        else:
            raise NotImplementedError(f"Scale factor {scale} not supported")
        super().__init__(*layers)


class EDSRSmall(nn.Module):
    """
    Lightweight EDSR variant (8 residual blocks, 64 feature channels).
    Maintains crisp edge transitions for solar filament structures without
    introducing batch-norm artifacts.
    """
    def __init__(self, scale_factor: int = 2, in_channels: int = 1,
                 num_features: int = 64, num_blocks: int = 8, res_scale: float = 0.1):
        super().__init__()
        self.scale_factor = scale_factor
        self.in_channels = in_channels

        # Head
        self.head = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)

        # Body (Residual blocks)
        body_blocks = [ResBlock(num_features, res_scale=res_scale) for _ in range(num_blocks)]
        body_blocks.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
        self.body = nn.Sequential(*body_blocks)

        # Upsampler
        self.upsampler = Upsampler(scale_factor, num_features)

        # Tail
        self.tail = nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.head(x)
        res = self.body(feat)
        res = res + feat
        up = self.upsampler(res)
        out = self.tail(up)
        return torch.clamp(out, 0.0, 1.0)
