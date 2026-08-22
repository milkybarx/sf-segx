"""
Mask2Former / Masked-Attention Transformer for Solar Filament Segmentation
==========================================================================
Implements Masked-attention Mask Transformer (Mask2Former) with support for:
1. Pretrained ResNet-34 Feature Pyramid Backbone (ImageNet pretrained weights)
2. Custom multi-scale convolutional encoder (from scratch)
3. Masked Cross-Attention Transformer Decoder (focuses strictly on filament regions)
4. High-resolution pixel mask feature generator
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import torchvision.models as models


class CustomPixelDecoder(nn.Module):
    """
    Custom 5-stage convolutional encoder with FPN.
    """
    def __init__(self, in_channels: int = 1, hidden_dim: int = 128):
        super().__init__()
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )  # 1/1 (512x512)

        self.conv_c2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )  # 1/2 (256x256)

        self.conv_c3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )  # 1/4 (128x128)

        self.conv_c4 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )  # 1/8 (64x64)

        self.conv_c5 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )  # 1/16 (32x32)

        self.lateral_c4 = nn.Conv2d(256, hidden_dim, 1)
        self.lateral_c3 = nn.Conv2d(128, hidden_dim, 1)
        self.lateral_c2 = nn.Conv2d(64, hidden_dim, 1)

        self.mask_features = nn.Sequential(
            nn.Conv2d(hidden_dim + 32, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        c1 = self.conv_c1(x)       # [B, 32, 512, 512]
        c2 = self.conv_c2(c1)      # [B, 64, 256, 256]
        c3 = self.conv_c3(c2)      # [B, 128, 128, 128]
        c4 = self.conv_c4(c3)      # [B, 256, 64, 64]
        c5 = self.conv_c5(c4)      # [B, 128, 32, 32]

        p5 = c5
        p4 = self.lateral_c4(c4) + F.interpolate(p5, size=c4.shape[2:], mode='bilinear', align_corners=False)
        p3 = self.lateral_c3(c3) + F.interpolate(p4, size=c3.shape[2:], mode='bilinear', align_corners=False)
        p2 = self.lateral_c2(c2) + F.interpolate(p3, size=c2.shape[2:], mode='bilinear', align_corners=False)

        p1 = F.interpolate(p2, size=c1.shape[2:], mode='bilinear', align_corners=False)
        mask_feat = self.mask_features(torch.cat([p1, c1], dim=1))  # [B, hidden_dim, 512, 512]

        return [p3, p4, p5], mask_feat


class ResNet34PixelDecoder(nn.Module):
    """
    Feature Pyramid Pixel Decoder powered by ImageNet-pretrained ResNet-34.
    """
    def __init__(self, in_channels: int = 1, hidden_dim: int = 128, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        backbone = models.resnet34(weights=weights)

        # Adapt input conv layer
        if in_channels == 1:
            orig_w = backbone.conv1.weight.data
            new_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            new_conv1.weight.data = orig_w.mean(dim=1, keepdim=True)
            self.conv1 = new_conv1
        elif in_channels == 3:
            self.conv1 = backbone.conv1
        else:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1  # [B, 64, 128, 128]
        self.layer2 = backbone.layer2  # [B, 128, 64, 64]
        self.layer3 = backbone.layer3  # [B, 256, 32, 32]
        self.layer4 = backbone.layer4  # [B, 512, 16, 16]

        # High-res stem for retaining original 512x512 sub-pixel spatial fidelity
        self.stem_c1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )

        # Lateral 1x1 projection convs
        self.lateral_c5 = nn.Conv2d(512, hidden_dim, 1)
        self.lateral_c4 = nn.Conv2d(256, hidden_dim, 1)
        self.lateral_c3 = nn.Conv2d(128, hidden_dim, 1)
        self.lateral_c2 = nn.Conv2d(64, hidden_dim, 1)

        # High-resolution mask feature generator
        self.mask_features = nn.Sequential(
            nn.Conv2d(hidden_dim + 32, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        c1 = self.stem_c1(x)  # [B, 32, 512, 512]

        x_stem = self.relu(self.bn1(self.conv1(x)))  # [B, 64, 256, 256]
        x_mp = self.maxpool(x_stem)                  # [B, 64, 128, 128]

        c2 = self.layer1(x_mp)  # [B, 64, 128, 128]  (1/4 scale)
        c3 = self.layer2(c2)    # [B, 128, 64, 64]   (1/8 scale)
        c4 = self.layer3(c3)    # [B, 256, 32, 32]   (1/16 scale)
        c5 = self.layer4(c4)    # [B, 512, 16, 16]   (1/32 scale)

        # Top-down FPN pathway
        p5 = self.lateral_c5(c5)
        p4 = self.lateral_c4(c4) + F.interpolate(p5, size=c4.shape[2:], mode='bilinear', align_corners=False)
        p3 = self.lateral_c3(c3) + F.interpolate(p4, size=c3.shape[2:], mode='bilinear', align_corners=False)
        p2 = self.lateral_c2(c2) + F.interpolate(p3, size=c2.shape[2:], mode='bilinear', align_corners=False)

        # Mask features at full 1/1 resolution
        p1 = F.interpolate(p2, size=c1.shape[2:], mode='bilinear', align_corners=False)
        mask_feat = self.mask_features(torch.cat([p1, c1], dim=1))  # [B, hidden_dim, 512, 512]

        return [p2, p3, p4], mask_feat


# Backwards compatibility alias
MultiScalePixelDecoder = CustomPixelDecoder


class MaskedCrossAttention(nn.Module):
    """
    Masked Cross-Attention: restricts cross-attention strictly to the foreground
    region of the predicted mask, ignoring solar surface noise.
    """
    def __init__(self, hidden_dim: int = 128, nheads: int = 8):
        super().__init__()
        self.nheads = nheads
        self.head_dim = hidden_dim // nheads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, queries: torch.Tensor, memory: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = queries.shape
        _, S, _ = memory.shape

        q = self.q_proj(queries).view(B, N, self.nheads, self.head_dim).transpose(1, 2)
        k = self.k_proj(memory).view(B, S, self.nheads, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory).view(B, S, self.nheads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            attn = attn + attn_mask

        attn_weights = F.softmax(attn, dim=-1)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        return self.out_proj(out)


class Mask2FormerDecoderLayer(nn.Module):
    """Single layer of Mask2Former Transformer Decoder."""
    def __init__(self, hidden_dim: int = 128, nheads: int = 8, dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = MaskedCrossAttention(hidden_dim, nheads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.self_attn = nn.MultiheadAttention(hidden_dim, nheads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, queries: torch.Tensor, memory: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q2 = self.cross_attn(queries, memory, attn_mask)
        queries = self.norm1(queries + self.dropout1(q2))

        q3, _ = self.self_attn(queries, queries, queries)
        queries = self.norm2(queries + self.dropout2(q3))

        q4 = self.linear2(self.dropout3(self.activation(self.linear1(queries))))
        queries = self.norm3(queries + self.dropout3(q4))
        return queries


class SubPixelBoundaryRefiner(nn.Module):
    """
    Sub-pixel boundary detail refinement module.
    Combines high-resolution 1/1 stem mask features with query mask logits to predict
    sharp residual boundary corrections for thin, faint chromospheric fibrils.
    """
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_dim + 1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, mask_feat: torch.Tensor, initial_logits: torch.Tensor) -> torch.Tensor:
        residual = self.refine(torch.cat([mask_feat, initial_logits], dim=1))
        return initial_logits + residual


class Mask2Former(nn.Module):
    """
    Complete Mask2Former architecture for Solar Filament Segmentation with Sub-Pixel Edge Refinement.
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_queries: int = 25,
        hidden_dim: int = 128,
        num_decoder_layers: int = 3,
        nheads: int = 8,
        backbone: str = 'resnet34',
        pretrained: bool = True,
        use_boundary_refiner: bool = True,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.backbone_name = backbone
        self.use_boundary_refiner = use_boundary_refiner

        # Select Pixel Decoder / Backbone
        if backbone.lower() in ('resnet34', 'resnet-34', 'resnet_34'):
            self.pixel_decoder = ResNet34PixelDecoder(in_channels, hidden_dim, pretrained=pretrained)
        else:
            self.pixel_decoder = CustomPixelDecoder(in_channels, hidden_dim)

        # Learnable Filament Query Embeddings
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Transformer Decoder Layers
        self.decoder_layers = nn.ModuleList([
            Mask2FormerDecoderLayer(hidden_dim=hidden_dim, nheads=nheads)
            for _ in range(num_decoder_layers)
        ])

        # Query projection for mask prediction
        self.mask_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Unified segmentation head
        self.head = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 1, 1),
        )

        # Sub-pixel boundary refinement
        if self.use_boundary_refiner:
            self.boundary_refiner = SubPixelBoundaryRefiner(hidden_dim)
        else:
            self.boundary_refiner = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # 1. Multi-scale feature extraction & pixel mask features
        multi_scale_features, mask_features = self.pixel_decoder(x)

        # 2. Initialize queries
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # [B, num_queries, hidden_dim]

        # 3. Iterative Transformer Decoding with Masked Attention
        attn_mask = None
        for i, layer in enumerate(self.decoder_layers):
            feat = multi_scale_features[i % len(multi_scale_features)]
            B_f, C_f, H_f, W_f = feat.shape
            memory = feat.flatten(2).transpose(1, 2)  # [B, H*W, hidden_dim]

            if attn_mask is not None and attn_mask.shape[-1] != (H_f * W_f):
                attn_mask = None

            queries = layer(queries, memory, attn_mask)

            mask_embed = self.mask_embed(queries)  # [B, num_queries, hidden_dim]
            feat_flat = feat.view(B_f, C_f, -1)
            pred_masks = torch.bmm(mask_embed, feat_flat).view(B, self.num_queries, H_f, W_f)

            mask_weights = (pred_masks.sigmoid() < 0.5).unsqueeze(1)
            attn_mask = torch.where(mask_weights.flatten(3), float('-1e4'), 0.0)

        # 4. Dense mask prediction using high-res mask features
        mask_embed = self.mask_embed(queries)
        combined_query = mask_embed.mean(dim=1).unsqueeze(-1).unsqueeze(-1)

        modulated_features = mask_features * combined_query
        initial_logits = self.head(modulated_features)

        # 5. Sub-pixel boundary detail refinement
        if self.boundary_refiner is not None:
            logits = self.boundary_refiner(mask_features, initial_logits)
        else:
            logits = initial_logits

        return logits

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def build_mask2former(config: dict = None) -> Mask2Former:
    """Build Mask2Former model."""
    if config is None:
        config = {}
    model = Mask2Former(
        in_channels=config.get('in_channels', 1),
        num_queries=config.get('num_queries', 20),
        hidden_dim=config.get('hidden_dim', 128),
        num_decoder_layers=config.get('num_decoder_layers', 3),
        backbone=config.get('backbone', 'resnet34'),
        pretrained=config.get('pretrained', True),
    )
    total, trainable = model.count_parameters()
    print(f"Mask2Former Model [{model.backbone_name.upper()}]: {total:,} total parameters ({trainable:,} trainable)")
    return model
