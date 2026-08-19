"""
Mask2Former / Masked-Attention Transformer for Solar Filament Segmentation
==========================================================================
Implements Masked-attention Mask Transformer (Mask2Former) for solar filament
detection and instance/semantic segmentation.

Key Advantages for Solar Imagery:
1. Masked Cross-Attention restricts attention to the foreground filament mask,
   completely ignoring solar chromospheric background noise.
2. High-resolution pixel decoder preserves fine, thread-like filament boundaries.
3. Multi-scale feature extraction across spatial resolutions.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional


class MultiScalePixelDecoder(nn.Module):
    """
    Feature Pyramid Pixel Decoder that extracts multi-scale feature maps
    at 1/4, 1/8, 1/16, and 1/32 resolutions.
    """
    def __init__(self, in_channels: int = 1, hidden_dim: int = 128):
        super().__init__()
        # Multi-scale convolutional encoder
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
        ) # 1/1 (512x512)

        self.conv_c2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        ) # 1/2 (256x256)

        self.conv_c3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        ) # 1/4 (128x128)

        self.conv_c4 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        ) # 1/8 (64x64)

        self.conv_c5 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        ) # 1/16 (32x32)

        # Lateral 1x1 convs to project to hidden_dim
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
        c1 = self.conv_c1(x)       # [B, 32, 512, 512]
        c2 = self.conv_c2(c1)      # [B, 64, 256, 256]
        c3 = self.conv_c3(c2)      # [B, 128, 128, 128]
        c4 = self.conv_c4(c3)      # [B, 256, 64, 64]
        c5 = self.conv_c5(c4)      # [B, 128, 32, 32]

        # Top-down FPN pathway
        p5 = c5
        p4 = self.lateral_c4(c4) + F.interpolate(p5, size=c4.shape[2:], mode='bilinear', align_corners=False)
        p3 = self.lateral_c3(c3) + F.interpolate(p4, size=c3.shape[2:], mode='bilinear', align_corners=False)
        p2 = self.lateral_c2(c2) + F.interpolate(p3, size=c2.shape[2:], mode='bilinear', align_corners=False)

        # Generate per-pixel mask features at original resolution
        p1 = F.interpolate(p2, size=c1.shape[2:], mode='bilinear', align_corners=False)
        mask_feat = self.mask_features(torch.cat([p1, c1], dim=1)) # [B, hidden_dim, 512, 512]

        return [p3, p4, p5], mask_feat


class ResNetPixelDecoder(nn.Module):
    """
    Feature Pyramid Pixel Decoder built on a real torchvision ResNet-34 backbone (in place
    of MultiScalePixelDecoder's from-scratch conv_c1-c5 stack), used by the "phase3" 768-res
    checkpoint (config: model.backbone="resnet34", model.pretrained=true). Submodule names
    (conv1/bn1/layer1-4) are assigned directly from torchvision's ResNet so the state_dict
    keys match torchvision's own convention exactly -- required for the trained checkpoint's
    weights to load, not just architectural taste.
    """
    def __init__(self, in_channels: int = 1, hidden_dim: int = 128, pretrained: bool = False):
        super().__init__()
        import torchvision

        resnet = torchvision.models.resnet34(
            weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        )
        # Adapt the stem for single-channel (grayscale) input instead of RGB.
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 1/4,  64ch
        self.layer2 = resnet.layer2  # 1/8,  128ch
        self.layer3 = resnet.layer3  # 1/16, 256ch
        self.layer4 = resnet.layer4  # 1/32, 512ch

        # Full-resolution stem (mirrors MultiScalePixelDecoder's conv_c1) purely for the
        # finest-detail skip connection into mask_features -- the ResNet stem above already
        # downsamples 4x before layer1, losing thin single-pixel filament threads otherwise.
        self.stem_c1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )

        self.lateral_c5 = nn.Conv2d(512, hidden_dim, 1)
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
        stem = self.stem_c1(x)  # [B, 32, H, W] full resolution

        h = self.relu(self.bn1(self.conv1(x)))
        h = self.maxpool(h)
        c2 = self.layer1(h)   # 1/4
        c3 = self.layer2(c2)  # 1/8
        c4 = self.layer3(c3)  # 1/16
        c5 = self.layer4(c4)  # 1/32

        p5 = self.lateral_c5(c5)
        p4 = self.lateral_c4(c4) + F.interpolate(p5, size=c4.shape[2:], mode='bilinear', align_corners=False)
        p3 = self.lateral_c3(c3) + F.interpolate(p4, size=c3.shape[2:], mode='bilinear', align_corners=False)
        p2 = self.lateral_c2(c2) + F.interpolate(p3, size=c2.shape[2:], mode='bilinear', align_corners=False)

        p1 = F.interpolate(p2, size=stem.shape[2:], mode='bilinear', align_corners=False)
        mask_feat = self.mask_features(torch.cat([p1, stem], dim=1))

        return [p3, p4, p5], mask_feat


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

        # [B, nheads, N, S]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            # Masked attention: -inf outside the predicted filament region
            attn = attn + attn_mask

        attn_weights = F.softmax(attn, dim=-1)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        return self.out_proj(out)


class Mask2FormerDecoderLayer(nn.Module):
    """Single layer of Mask2Former Transformer Decoder."""
    def __init__(self, hidden_dim: int = 128, nheads: int = 8, dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        # 1. Masked Cross-Attention
        self.cross_attn = MaskedCrossAttention(hidden_dim, nheads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)

        # 2. Self-Attention between filament queries
        self.self_attn = nn.MultiheadAttention(hidden_dim, nheads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)

        # 3. FFN
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, queries: torch.Tensor, memory: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Cross-Attention with Mask
        q2 = self.cross_attn(queries, memory, attn_mask)
        queries = self.norm1(queries + self.dropout1(q2))

        # Self-Attention
        q3, _ = self.self_attn(queries, queries, queries)
        queries = self.norm2(queries + self.dropout2(q3))

        # FFN
        q4 = self.linear2(self.dropout3(self.activation(self.linear1(queries))))
        queries = self.norm3(queries + self.dropout3(q4))

        return queries


class Mask2Former(nn.Module):
    """
    Complete Mask2Former architecture for Solar Filament Segmentation.
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_queries: int = 20, # Up to 20 filament instances per solar observation
        hidden_dim: int = 128,
        num_decoder_layers: int = 3,
        nheads: int = 8,
        backbone: str = "scratch",  # "scratch" (from-scratch conv stack) or "resnet34"
        pretrained: bool = False,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        # Pixel Decoder (Feature Pyramid)
        if backbone == "resnet34":
            self.pixel_decoder = ResNetPixelDecoder(in_channels, hidden_dim, pretrained=pretrained)
        else:
            self.pixel_decoder = MultiScalePixelDecoder(in_channels, hidden_dim)

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

        # Final unified segmentation head
        self.head = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # 1. Multi-scale feature extraction & pixel mask features
        multi_scale_features, mask_features = self.pixel_decoder(x)

        # 2. Initialize queries
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1) # [B, num_queries, hidden_dim]

        # 3. Iterative Transformer Decoding with Masked Attention
        attn_mask = None
        for i, layer in enumerate(self.decoder_layers):
            # Select multi-scale memory feature map
            feat = multi_scale_features[i % len(multi_scale_features)]
            B_f, C_f, H_f, W_f = feat.shape
            memory = feat.flatten(2).transpose(1, 2) # [B, H*W, hidden_dim]

            # Adjust previous predicted mask to current layer spatial resolution
            if attn_mask is not None and attn_mask.shape[-1] != (H_f * W_f):
                attn_mask = None # Reset when switching feature scale

            # Apply layer
            queries = layer(queries, memory, attn_mask)

            # Predict intermediate mask for next layer's masked attention
            mask_embed = self.mask_embed(queries) # [B, num_queries, hidden_dim]
            feat_flat = feat.view(B_f, C_f, -1)
            pred_masks = torch.bmm(mask_embed, feat_flat).view(B, self.num_queries, H_f, W_f)

            # Next layer masked cross-attention: mask out background
            mask_weights = (pred_masks.sigmoid() < 0.5).unsqueeze(1) # [B, 1, num_queries, H_f, W_f]
            attn_mask = torch.where(mask_weights.flatten(3), float('-1e4'), 0.0)

        # 4. Final dense mask prediction using high-res mask features
        mask_embed = self.mask_embed(queries) # [B, num_queries, hidden_dim]
        # Aggregate top filament queries: [B, hidden_dim, 1, 1]
        combined_query = mask_embed.mean(dim=1).unsqueeze(-1).unsqueeze(-1)
        
        # Spatial channel modulation of high-res mask features
        modulated_features = mask_features * combined_query
        logits = self.head(modulated_features)

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
        backbone=config.get('backbone', 'scratch'),
        # Loading our own trained checkpoint immediately overwrites every weight anyway --
        # pretrained=False here regardless of the checkpoint's own training-time config,
        # so inference never depends on an ImageNet download.
        pretrained=False,
    )
    total, trainable = model.count_parameters()
    print(f"Mask2Former Model: {total:,} total parameters ({trainable:,} trainable)")
    return model
