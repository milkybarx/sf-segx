"""
FSRCNN: Fast Super-Resolution Convolutional Neural Network
==========================================================
Reference: Dong et al., "Accelerating the Super-Resolution Convolutional
Neural Network", ECCV 2016.
"""

import math
import torch
import torch.nn as nn


class FSRCNN(nn.Module):
    """
    Fast Super-Resolution CNN with an hourglass design and transposed convolution.
    """
    def __init__(self, scale_factor: int = 2, in_channels: int = 1,
                 d: int = 56, s: int = 12, m: int = 4):
        super().__init__()
        self.scale_factor = scale_factor
        self.in_channels = in_channels

        # 1. Feature extraction
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels, d, kernel_size=5, padding=2),
            nn.PReLU(d)
        )

        # 2. Shrinking
        self.shrinking = nn.Sequential(
            nn.Conv2d(d, s, kernel_size=1),
            nn.PReLU(s)
        )

        # 3. Non-linear mapping
        mapping_layers = []
        for _ in range(m):
            mapping_layers.append(nn.Conv2d(s, s, kernel_size=3, padding=1))
            mapping_layers.append(nn.PReLU(s))
        self.mapping = nn.Sequential(*mapping_layers)

        # 4. Expanding
        self.expanding = nn.Sequential(
            nn.Conv2d(s, d, kernel_size=1),
            nn.PReLU(d)
        )

        # 5. Deconvolution (upsampling)
        self.deconv = nn.ConvTranspose2d(
            d, in_channels, kernel_size=9, stride=scale_factor,
            padding=4, output_padding=scale_factor - 1
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.feature_extraction(x)
        out = self.shrinking(out)
        out = self.mapping(out)
        out = self.expanding(out)
        out = self.deconv(out)
        return torch.clamp(out, 0.0, 1.0)
