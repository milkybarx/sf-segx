"""
ESPCN: Efficient Sub-Pixel Convolutional Neural Network
======================================================
Reference: Shi et al., "Real-Time Single Image and Video Super-Resolution Using
an Efficient Sub-Pixel Convolutional Neural Network", CVPR 2016.
"""

import math
import torch
import torch.nn as nn


class ESPCN(nn.Module):
    """
    Lightweight Sub-Pixel Convolutional Network for Fast Solar Crop Upscaling.
    Operates entirely in LR feature space before upsampling at the final layer.
    """
    def __init__(self, scale_factor: int = 2, in_channels: int = 1, hidden_channels: int = 64):
        super().__init__()
        self.scale_factor = scale_factor
        self.in_channels = in_channels

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(hidden_channels // 2, in_channels * (scale_factor ** 2), kernel_size=3, padding=1),
        )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('tanh'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Custom initialization for sub-pixel layer to act as bilinear upscale initially
        last_conv = self.feature_extractor[-1]
        nn.init.xavier_normal_(last_conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.feature_extractor(x)
        out = self.pixel_shuffle(out)
        return torch.clamp(out, 0.0, 1.0)
