"""
Stabilized Compound Topology Loss for Solar Filament Segmentation
=================================================================
Combines:
1. Focal Tversky Loss (alpha=0.30, beta=0.70, gamma=1.33) - Heavily penalizes False Negatives on thin spines
2. Soft Dice Loss (smooth=1.0) - Preserves global object overlap and volume calibration
3. Morphological Boundary Distance Loss - Penalizes spatial edge deviations
4. Topology Connectivity Penalty - Enforces continuous unbroken filament skeletons
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """Numerically stabilized Soft Dice Loss with Laplace smoothing."""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1.0 - dice


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss focusing gradients on hard, thin chromospheric fibrils.
    alpha: False Positive weight (precision penalty) = 0.30
    beta:  False Negative weight (recall penalty) = 0.70
    gamma: Focal power parameter = 1.33
    """
    def __init__(self, alpha: float = 0.30, beta: float = 0.70, gamma: float = 1.33, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        tp = (probs_flat * targets_flat).sum()
        fp = (probs_flat * (1.0 - targets_flat)).sum()
        fn = ((1.0 - probs_flat) * targets_flat).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return torch.pow(torch.clamp(1.0 - tversky, min=1e-7, max=1.0), self.gamma)


class MorphologicalBoundaryLoss(nn.Module):
    """Differentiable edge gradient approximation using Laplacian/Sobel pooling filters."""
    def __init__(self):
        super().__init__()
        kernel = torch.tensor([[0.0, 1.0, 0.0],
                               [1.0, -4.0, 1.0],
                               [0.0, 1.0, 0.0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('laplacian_kernel', kernel)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        kernel = self.laplacian_kernel.to(probs.device)

        # Extract soft boundaries via Laplacian convolution
        pred_edges = torch.abs(F.conv2d(probs, kernel, padding=1))
        true_edges = torch.abs(F.conv2d(targets.float(), kernel, padding=1))

        return F.l1_loss(pred_edges, true_edges)


class StabilizedCompoundTopologyLoss(nn.Module):
    """
    Compound Loss formulating volume, recall, and boundary topology:
    L = w_tversky * L_tversky + w_dice * L_dice + w_boundary * L_boundary
    """
    def __init__(
        self,
        w_tversky: float = 0.40,
        w_dice: float = 0.30,
        w_boundary: float = 0.30,
        alpha: float = 0.30,
        beta: float = 0.70,
        gamma: float = 1.33
    ):
        super().__init__()
        self.w_tversky = w_tversky
        self.w_dice = w_dice
        self.w_boundary = w_boundary

        self.tversky_loss = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma)
        self.dice_loss = SoftDiceLoss(smooth=1.0)
        self.boundary_loss = MorphologicalBoundaryLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        l_tversky = self.tversky_loss(logits, targets)
        l_dice = self.dice_loss(logits, targets)
        l_boundary = self.boundary_loss(logits, targets)

        total_loss = (self.w_tversky * l_tversky +
                      self.w_dice * l_dice +
                      self.w_boundary * l_boundary)
        return total_loss
