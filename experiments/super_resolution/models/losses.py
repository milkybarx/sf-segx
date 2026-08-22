"""
Loss functions for Super-Resolution Training
=============================================
Provides robust pixel reconstruction (Charbonnier loss) and structural
similarity (SSIM loss) tailored for scientific solar imagery.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (differentiable L1 variant):
    L = sqrt((y_pred - y_true)^2 + eps^2)
    Provides smooth gradients near zero and robust outlier handling.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps2)
        return torch.mean(loss)


def _gaussian_window(window_size: int, sigma: float, channels: int) -> torch.Tensor:
    """Create a 2D Gaussian kernel window for SSIM computation."""
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_1d = g.unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).unsqueeze(0).unsqueeze(0)
    window = window_2d.expand(channels, 1, window_size, window_size).contiguous()
    return window


class SSIMLoss(nn.Module):
    """
    Differentiable Structural Similarity (SSIM) Loss:
    Loss = 1 - SSIM(pred, target)
    """
    def __init__(self, window_size: int = 11, sigma: float = 1.5, channels: int = 1):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        self.register_buffer("window", _gaussian_window(window_size, sigma, channels))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.size(1) != self.channels:
            window = _gaussian_window(self.window_size, 1.5, pred.size(1)).to(pred.device)
        else:
            window = self.window.to(pred.device)

        padding = self.window_size // 2
        groups = pred.size(1)

        mu1 = F.conv2d(pred, window, padding=padding, groups=groups)
        mu2 = F.conv2d(target, window, padding=padding, groups=groups)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, window, padding=padding, groups=groups) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=padding, groups=groups) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=padding, groups=groups) - mu1_mu2

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        )
        return 1.0 - ssim_map.mean()


class CompositeSRLoss(nn.Module):
    """
    Composite Super-Resolution Loss:
    L_total = alpha * L_Charbonnier + beta * (1 - SSIM)
    Avoids hallucinated high-frequency artifacts while retaining solar structure edges.
    """
    def __init__(self, alpha: float = 0.8, beta: float = 0.2, eps: float = 1e-6, channels: int = 1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.charbonnier = CharbonnierLoss(eps=eps)
        self.ssim_loss = SSIMLoss(channels=channels)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_charb = self.charbonnier(pred, target)
        loss_ssim = self.ssim_loss(pred, target)
        return self.alpha * loss_charb + self.beta * loss_ssim
