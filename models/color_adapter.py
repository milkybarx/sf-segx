"""
Color -> H-alpha style adapter.

Every model in this repo (Mask2Former, SegFormer, U-Net, DeepLabV3+, Attention U-Net) was
trained exclusively on MAGFiLO H-alpha imagery, which is genuinely single-channel (verified:
every training JPEG has R==G==B exactly, just saved as a 3-channel file). None of them have
ever seen a real color image, so naive RGB->grayscale averaging (cv2.COLOR_BGR2GRAY, a fixed
0.299R+0.587G+0.114B weighting) is not guaranteed to preserve filament contrast for an
arbitrarily color-graded or false-color input -- a hue rotation can shift which channel
carries the true luminance structure.

ColorToHAlphaNet is a small U-Net (3-channel RGB -> 1-channel) trained with *synthetic*
colorization augmentation (see scripts/train_color_adapter.py): every training example is
built by re-coloring a real H-alpha grayscale image with a random hue/saturation/gamma
transform, with the network's target being the original untouched grayscale. Since the
dataset has no real color solar imagery, this self-supervised setup is the only training
signal available -- but because the augmentation spans a wide range of global tints (not just
the specific look of any one instrument), the network learns a general "recover the
luminance/contrast structure regardless of color cast" mapping rather than memorizing one
specific color scheme, which is what lets it generalize to unseen colored inputs at inference.
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ColorToHAlphaNet(nn.Module):
    """3-channel RGB in, 1-channel H-alpha-style grayscale out (sigmoid, [0,1])."""

    def __init__(self, base: int = 24):
        super().__init__()
        self.enc1 = ConvBlock(3, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base * 2, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)
        self.out_conv = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.out_conv(d1))


def build_color_adapter(base: int = 24) -> ColorToHAlphaNet:
    return ColorToHAlphaNet(base=base)
