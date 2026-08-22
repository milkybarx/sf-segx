"""
U-Net Segmentation Model
=========================
Standard U-Net architecture for solar filament segmentation.
Encoder-decoder with skip connections, batch normalization, and configurable depth.
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two consecutive conv-bn-relu blocks."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for binary segmentation.

    Architecture:
        Encoder: 4 downsampling blocks with max pooling
        Bottleneck: Deepest feature extraction
        Decoder: 4 upsampling blocks with skip connections
        Output: 1x1 conv + sigmoid
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: list = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upconvs = nn.ModuleList()

        # Encoder path
        ch = in_channels
        for f in features:
            self.encoder_blocks.append(DoubleConv(ch, f, dropout=dropout))
            ch = f

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, dropout=dropout)

        # Decoder path
        for f in reversed(features):
            self.upconvs.append(
                nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
            )
            self.decoder_blocks.append(DoubleConv(f * 2, f, dropout=dropout))

        # Final output
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        skip_connections = []
        for encoder in self.encoder_blocks:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        skip_connections = skip_connections[::-1]
        for idx in range(len(self.decoder_blocks)):
            x = self.upconvs[idx](x)
            skip = skip_connections[idx]

            # Handle size mismatch from odd dimensions
            if x.shape != skip.shape:
                x = nn.functional.interpolate(
                    x, size=skip.shape[2:], mode='bilinear', align_corners=True
                )

            x = torch.cat([skip, x], dim=1)
            x = self.decoder_blocks[idx](x)

        # Output probability map
        logits = self.final_conv(x)
        return logits  # Raw logits — sigmoid applied in loss or inference

    def predict(self, x):
        """Forward pass with sigmoid for inference."""
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def build_unet(config: dict = None) -> UNet:
    """Build U-Net from config dict."""
    if config is None:
        config = {}

    model = UNet(
        in_channels=config.get('in_channels', 1),
        out_channels=config.get('out_channels', 1),
        features=config.get('features', [64, 128, 256, 512]),
        dropout=config.get('dropout', 0.1),
    )

    total, trainable = model.count_parameters()
    print(f"U-Net: {total:,} total params, {trainable:,} trainable")

    return model
