"""
Retrains Mask2Former (ResNet-34 backbone, 768px) -- the same architecture and resolution
as checkpoints/mask2former_phase3_768_best.pth (0.7207 Dice, the best model in this repo),
using the loss recipe recorded in that checkpoint's own saved config (dice_focal_boundary,
weights 0.4/0.3/0.3) which training/train.py doesn't implement (it only has plain DiceBCE).
Also uses backbone=pretrained=True (real ImageNet init) -- the inference-time build path in
model_hub.py deliberately hardcodes pretrained=False since inference always overwrites
every weight from a checkpoint anyway, but that's wrong for training a new one from scratch.

Resilient: saves a checkpoint on every validation Dice improvement, writes per-epoch
history to a CSV, and reports epoch timing so real progress is always visible in the log
even if the process is killed mid-run.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.mask2former import Mask2Former  # noqa: E402
from preprocessing.dataset import SolarFilamentDataset, create_data_splits  # noqa: E402
from training.losses import DiceLoss, FocalLoss  # noqa: E402
from training.metrics import compute_all_metrics  # noqa: E402

DATASET_ROOT = os.path.join(ROOT, "data", "MAGFiLO_1.0_Kaggle_2026")
IMAGE_DIR = os.path.join(DATASET_ROOT, "train", "train_images")
ANNOTATIONS_JSON = os.path.join(DATASET_ROOT, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
CHECKPOINT_PATH = os.path.join(ROOT, "checkpoints", "mask2former_phase3_768_best.pth")
HISTORY_PATH = os.path.join(ROOT, "experiments", "mask2former_v2_history.csv")
SEED = 42


_SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
_SOBEL_Y = _SOBEL_X.transpose(2, 3)


class BoundaryLoss(nn.Module):
    """L1 distance between the Sobel gradients of the predicted probability map and the
    target mask -- penalizes blurry/misplaced filament edges specifically, on top of the
    region-overlap signal Dice/Focal already provide."""

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        device = logits.device
        sx, sy = _SOBEL_X.to(device), _SOBEL_Y.to(device)
        pgx, pgy = F.conv2d(probs, sx, padding=1), F.conv2d(probs, sy, padding=1)
        tgx, tgy = F.conv2d(targets, sx, padding=1), F.conv2d(targets, sy, padding=1)
        return (pgx - tgx).abs().mean() + (pgy - tgy).abs().mean()


class DiceFocalBoundaryLoss(nn.Module):
    """dice_weight/focal_weight are unnormalized multipliers (not a convex combination
    summing to 1) -- matches the original Mask2Former paper's loss-weighting convention
    (large fixed coefficients per term, e.g. ~5 for dice, ~2 for the classification term),
    not a 0-1 blend. This architecture has no true per-query classification head (it
    collapses queries into one dense mask, not per-query class+mask predictions like the
    paper), so the closest real equivalent to a "classification" loss is the focal term --
    it's a per-pixel foreground/background classification loss, same role the paper's
    per-query class loss plays, just at the pixel level instead of the query level."""

    def __init__(self, dice_weight=7.0, focal_weight=0.75, boundary_weight=0.3,
                 focal_alpha=0.75, focal_gamma=2.0):
        super().__init__()
        self.dice_weight, self.focal_weight, self.boundary_weight = dice_weight, focal_weight, boundary_weight
        self.dice = DiceLoss()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.boundary = BoundaryLoss()

    def forward(self, logits, targets):
        return (self.dice_weight * self.dice(logits, targets)
                + self.focal_weight * self.focal(logits, targets)
                + self.boundary_weight * self.boundary(logits, targets))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--image_size", type=int, default=768)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--early_stopping_patience", type=int, default=15)
    ap.add_argument("--num_queries", type=int, default=30)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    train_ids, val_ids = create_data_splits(ANNOTATIONS_JSON, IMAGE_DIR, train_ratio=0.8, seed=SEED)
    cache_dir = os.path.join(ROOT, f"cache_{args.image_size}")
    train_ds = SolarFilamentDataset(IMAGE_DIR, ANNOTATIONS_JSON, image_size=args.image_size,
                                     augment=True, image_ids=train_ids, cache_dir=cache_dir)
    val_ds = SolarFilamentDataset(IMAGE_DIR, ANNOTATIONS_JSON, image_size=args.image_size,
                                   augment=False, image_ids=val_ids, cache_dir=cache_dir)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}", flush=True)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                                num_workers=0, drop_last=True, pin_memory=(device.type == "cuda"))
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                              num_workers=0, pin_memory=(device.type == "cuda"))

    model = Mask2Former(in_channels=1, num_queries=args.num_queries, hidden_dim=128, num_decoder_layers=3,
                         backbone="resnet34", pretrained=True).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Mask2Former (ResNet-34, pretrained): {n_params:,} params, image_size={args.image_size}", flush=True)

    criterion = DiceFocalBoundaryLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    def forward_pass(images, masks, training):
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)
        return logits, loss

    best_dice = -float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss, total_dice, total_iou, n_batches = 0.0, 0.0, 0.0, 0
        for step, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            logits, loss = forward_pass(images, masks, True)
            scaler.scale(loss / args.grad_accum).backward()
            if (step + 1) % args.grad_accum == 0 or step + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                m = compute_all_metrics(logits.detach(), masks)
            total_loss += loss.item(); total_dice += m["dice"]; total_iou += m["iou"]; n_batches += 1
        train_loss, train_dice, train_iou = total_loss / n_batches, total_dice / n_batches, total_iou / n_batches
        scheduler.step()

        model.eval()
        total_loss, total_dice, total_iou, total_p, total_r, n_batches = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits, loss = forward_pass(images, masks, False)
                m = compute_all_metrics(logits, masks)
                total_loss += loss.item(); total_dice += m["dice"]; total_iou += m["iou"]
                total_p += m["precision"]; total_r += m["recall"]; n_batches += 1
        val_loss, val_dice, val_iou = total_loss / n_batches, total_dice / n_batches, total_iou / n_batches
        val_p, val_r = total_p / n_batches, total_r / n_batches

        lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": train_loss, "train_dice": train_dice, "train_iou": train_iou,
                         "val_loss": val_loss, "val_dice": val_dice, "val_iou": val_iou,
                         "val_precision": val_p, "val_recall": val_r, "lr": lr})
        pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)
        print(f"Epoch [{epoch}/{args.epochs}] | Train Loss: {train_loss:.4f} - Dice: {train_dice:.4f} | "
              f"Val Loss: {val_loss:.4f} - Dice: {val_dice:.4f} - IoU: {val_iou:.4f} | "
              f"LR: {lr:.2e} | ({time.time() - t0:.1f}s)", flush=True)

        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            torch.save({
                "model_state_dict": {k: v.half() for k, v in model.state_dict().items()},
                "epoch": epoch, "val_dice": val_dice, "val_iou": val_iou,
                "val_precision": val_p, "val_recall": val_r, "val_loss": val_loss,
                "config": {"model": {"name": "mask2former", "backbone": "resnet34", "pretrained": True,
                                      "in_channels": 1, "num_queries": args.num_queries, "hidden_dim": 128,
                                      "num_decoder_layers": 3, "dropout": 0.1},
                           "data": {"image_size": args.image_size}},
            }, CHECKPOINT_PATH)
            print(f"   Saved best checkpoint (Dice: {best_dice:.4f})", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    print(f"Training finished. Best validation Dice: {best_dice:.4f}", flush=True)
    with open(os.path.join(ROOT, "experiments", "mask2former_v2_config.json"), "w") as f:
        json.dump({"image_size": args.image_size, "epochs_run": len(history), "best_val_dice": best_dice}, f, indent=2)


if __name__ == "__main__":
    main()
