"""
Mask2Former v3 -- applies a specific requested recipe on top of v2
(scripts/train_mask2former_v2.py), tried because v2 (784px, dice+focal+boundary, num_queries=30)
was progressing well but far too slow on this machine (system RAM was nearly exhausted by other
running apps, ~4.5x slower than a clean-system calibration run) to finish in a reasonable time.

Changes from v2, all per explicit request:
  1. num_queries lowered to 25 (was 30/20).
  2. 3-channel input instead of flat grayscale-replicated-to-3: channel 0 is the existing
     enhanced/normalized grayscale, channel 1 is a multi-scale Hessian ridge-vesselness
     response (classical/hessian.py -- filaments are thin ridge-like structures, so this is
     a real, targeted signal, not filler), channel 2 is the limb-corrected-but-not-CLAHE'd
     view. All three are real, different transforms of the same image, not 3 copies of one.
  3. A stricter 88% (vs the shared default 93%) solar-disk radius mask for this run only --
     preprocessing.solar_preprocessor.SolarPreprocessor.disk_shrink was made a constructor
     parameter specifically so this doesn't change the default the currently-deployed
     checkpoint's own inference relies on.
  4. Loss replaced with StabilizedCompoundTopologyLoss: there is no citable paper/library
     actually named this -- this implements the literal description (a stabilized, compound,
     topology-aware loss): Dice + Focal ("classification"/per-pixel term) + boundary
     (Sobel-gradient) + a clDice-style soft-skeleton topology term, each individually value-
     clamped for numerical stability, with Dice/Focal computed on an importance-sampled point
     set (25000 points/image, 70% random + 30% boundary-biased) rather than every pixel --
     the same point-based mask loss idea the real Mask2Former paper uses (its own
     train_num_points, default 12544 there), applied here at the requested 25000.
  5. Post-training threshold sweep over [0.24, 0.50].
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from classical.hessian import multiscale_hessian_response  # noqa: E402
from models.mask2former import Mask2Former  # noqa: E402
from preprocessing.dataset import coco_poly_to_mask, create_data_splits, load_coco_annotations  # noqa: E402
from preprocessing.solar_preprocessor import SolarPreprocessor  # noqa: E402
from training.losses import DiceLoss, FocalLoss  # noqa: E402
from training.metrics import compute_all_metrics  # noqa: E402

DATASET_ROOT = os.path.join(ROOT, "data", "MAGFiLO_1.0_Kaggle_2026")
IMAGE_DIR = os.path.join(DATASET_ROOT, "train", "train_images")
ANNOTATIONS_JSON = os.path.join(DATASET_ROOT, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
CHECKPOINT_PATH = os.path.join(ROOT, "checkpoints", "mask2former_v3_best.pth")
HISTORY_PATH = os.path.join(ROOT, "experiments", "mask2former_v3_history.csv")
SEED = 42
DISK_SHRINK = 0.88


class ThreeChannelDataset(Dataset):
    """channel0=enhanced grayscale, channel1=multiscale Hessian ridge response (normalized),
    channel2=limb-corrected-only view (no CLAHE) -- three genuinely different transforms of
    the same H-alpha image, not 3 copies of one."""

    def __init__(self, image_ids, images_dict, annotations_by_image, image_size, cache_dir,
                 augment=False):
        self.image_ids = image_ids
        self.images_dict = images_dict
        self.annotations_by_image = annotations_by_image
        self.image_size = image_size
        self.cache_dir = cache_dir
        self.augment = augment
        self.preprocessor = SolarPreprocessor(target_size=image_size, disk_shrink=DISK_SHRINK)
        os.makedirs(cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.image_ids)

    def _generate_mask(self, image_id, height, width):
        mask = np.zeros((height, width), dtype=np.uint8)
        for ann in self.annotations_by_image.get(image_id, []):
            seg = ann.get("segmentation", [])
            if seg:
                mask = np.maximum(mask, coco_poly_to_mask(seg, height, width))
        return mask

    def _build_3channel(self, raw_gray, cx, cy, radius, disk_mask):
        corrected = self.preprocessor.correct_limb_darkening(raw_gray, cx, cy, radius)
        normalized = self.preprocessor.normalize(corrected, disk_mask)
        denoised = self.preprocessor.denoise(normalized, sigma=1.0)
        enhanced = self.preprocessor.enhance_contrast(denoised, clip_limit=2.0)
        enhanced[disk_mask == 0] = 0

        ridge = multiscale_hessian_response(enhanced, scales=[1, 2, 3])
        ridge = ridge / (ridge.max() + 1e-6)
        ridge = (ridge * 255).astype(np.uint8)
        ridge[disk_mask == 0] = 0

        limb_only = corrected.copy()
        limb_only[disk_mask == 0] = 0

        stacked = np.stack([enhanced, ridge, limb_only], axis=-1)  # HWC uint8
        return cv2.resize(stacked, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.images_dict[image_id]
        file_name = img_info["file_name"]
        base_id = os.path.splitext(file_name)[0]
        cached_img = os.path.join(self.cache_dir, f"{base_id}_img3.npy")
        cached_mask = os.path.join(self.cache_dir, f"{base_id}_mask.npy")

        if os.path.exists(cached_img) and os.path.exists(cached_mask):
            image3 = np.load(cached_img)
            mask_float = np.load(cached_mask)
        else:
            raw = cv2.imread(os.path.join(IMAGE_DIR, file_name), cv2.IMREAD_GRAYSCALE)
            orig_h, orig_w = raw.shape[:2]
            raw_mask = self._generate_mask(image_id, orig_h, orig_w)
            cx, cy, radius = self.preprocessor.detect_solar_disk(raw)
            disk_mask = self.preprocessor.create_disk_mask(raw.shape, cx, cy, radius)
            image3 = self._build_3channel(raw, cx, cy, radius, disk_mask)
            mask_resized = cv2.resize(raw_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            mask_float = mask_resized.astype(np.float32)
            np.save(cached_img, image3)
            np.save(cached_mask, mask_float)

        if self.augment:
            if np.random.random() > 0.5:
                image3 = np.fliplr(image3).copy(); mask_float = np.fliplr(mask_float).copy()
            if np.random.random() > 0.5:
                image3 = np.flipud(image3).copy(); mask_float = np.flipud(mask_float).copy()
            k = np.random.randint(0, 4)
            if k:
                image3 = np.rot90(image3, k).copy(); mask_float = np.rot90(mask_float, k).copy()

        image_tensor = torch.from_numpy(image3.astype(np.float32) / 255.0).permute(2, 0, 1)  # [3,H,W]
        mask_tensor = torch.from_numpy(mask_float).unsqueeze(0)  # [1,H,W]
        return image_tensor, mask_tensor


_SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
_SOBEL_Y = _SOBEL_X.transpose(2, 3)


def soft_skeletonize(x, iters=5):
    """Differentiable soft skeleton via iterative min/max pooling (clDice-style)."""
    for _ in range(iters):
        min_pool = -F.max_pool2d(-x, 3, stride=1, padding=1)
        contour = F.relu(F.max_pool2d(min_pool, 3, stride=1, padding=1) - min_pool)
        x = F.relu(x - contour)
    return x


class StabilizedCompoundTopologyLoss(nn.Module):
    def __init__(self, dice_weight=9.0, focal_weight=1.0, boundary_weight=0.4, topology_weight=0.3,
                 focal_alpha=0.75, focal_gamma=2.0, num_points=25000, eps=1e-6):
        super().__init__()
        self.dice_weight, self.focal_weight = dice_weight, focal_weight
        self.boundary_weight, self.topology_weight = boundary_weight, topology_weight
        self.dice = DiceLoss()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.num_points = num_points
        self.eps = eps

    def _sample_points(self, logits, targets):
        """Importance-sample points per image (70% uniform + 30% boundary-biased) without any
        host<->device sync: .nonzero()/.numel()-gated branches each force the GPU to block and
        report a value back to Python, which measured as the dominant per-step cost (~11s of an
        ~11.5s "backward" measurement was actually this, not the model). torch.multinomial
        samples according to a per-pixel weight distribution entirely on-device instead."""
        B, _, H, W = logits.shape
        device = logits.device
        n_boundary = int(self.num_points * 0.3)
        n_random = self.num_points - n_boundary

        boundary_map = (F.max_pool2d(targets, 3, stride=1, padding=1)
                         - (-F.max_pool2d(-targets, 3, stride=1, padding=1)))  # [B,1,H,W], >0 at edges
        boundary_weights = boundary_map.view(B, -1).clamp(min=0) + 1e-6  # never all-zero

        flat_logits = logits.view(B, -1)
        flat_targets = targets.view(B, -1)

        random_idx = torch.randint(0, H * W, (B, n_random), device=device)
        boundary_idx = torch.multinomial(boundary_weights, n_boundary, replacement=True)
        idx = torch.cat([random_idx, boundary_idx], dim=1)  # [B, num_points]

        sampled_logits = torch.gather(flat_logits, 1, idx)
        sampled_targets = torch.gather(flat_targets, 1, idx)
        return sampled_logits, sampled_targets

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        boundary = self._boundary_term(probs, targets)
        topology = self._topology_term(probs, targets)

        sampled_logits, sampled_targets = self._sample_points(logits, targets)
        dice = self.dice(sampled_logits, sampled_targets)
        focal = self.focal(sampled_logits, sampled_targets)

        total = (self.dice_weight * dice.clamp(0, 2)
                 + self.focal_weight * focal.clamp(0, 5)
                 + self.boundary_weight * boundary.clamp(0, 5)
                 + self.topology_weight * topology.clamp(0, 2))
        return total

    def _boundary_term(self, probs, targets):
        device = probs.device
        sx, sy = _SOBEL_X.to(device), _SOBEL_Y.to(device)
        pgx, pgy = F.conv2d(probs, sx, padding=1), F.conv2d(probs, sy, padding=1)
        tgx, tgy = F.conv2d(targets, sx, padding=1), F.conv2d(targets, sy, padding=1)
        return (pgx - tgx).abs().mean() + (pgy - tgy).abs().mean()

    def _topology_term(self, probs, targets):
        skel_pred = soft_skeletonize(probs)
        skel_true = soft_skeletonize(targets)
        t_prec = (skel_pred * targets).sum() / (skel_pred.sum() + self.eps)
        t_sens = (skel_true * probs).sum() / (skel_true.sum() + self.eps)
        cl_dice = 2 * t_prec * t_sens / (t_prec + t_sens + self.eps)
        return 1.0 - cl_dice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--image_size", type=int, default=768)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num_queries", type=int, default=25)
    ap.add_argument("--early_stopping_patience", type=int, default=15)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"num_queries={args.num_queries}  disk_shrink={DISK_SHRINK}  channels=3 "
          f"(enhanced, hessian-ridge, limb-corrected)", flush=True)

    images_dict, annotations_by_image, _ = load_coco_annotations(ANNOTATIONS_JSON)
    train_ids, val_ids = create_data_splits(ANNOTATIONS_JSON, IMAGE_DIR, train_ratio=0.8, seed=SEED)
    cache_dir = os.path.join(ROOT, f"cache_{args.image_size}_3ch")

    train_ds = ThreeChannelDataset(train_ids, images_dict, annotations_by_image, args.image_size, cache_dir, augment=True)
    val_ds = ThreeChannelDataset(val_ids, images_dict, annotations_by_image, args.image_size, cache_dir, augment=False)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
                               drop_last=True, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
                             pin_memory=(device.type == "cuda"))

    model = Mask2Former(in_channels=3, num_queries=args.num_queries, hidden_dim=128, num_decoder_layers=3,
                         backbone="resnet34", pretrained=True).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Mask2Former v3 (3ch, ResNet-34, pretrained): {n_params:,} params, image_size={args.image_size}", flush=True)

    criterion = StabilizedCompoundTopologyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    def forward_pass(images, masks):
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
            logits, loss = forward_pass(images, masks)
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
                logits, loss = forward_pass(images, masks)
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
                                      "in_channels": 3, "num_queries": args.num_queries, "hidden_dim": 128,
                                      "num_decoder_layers": 3, "dropout": 0.1},
                           "data": {"image_size": args.image_size, "disk_shrink": DISK_SHRINK}},
            }, CHECKPOINT_PATH)
            print(f"   Saved best checkpoint (Dice: {best_dice:.4f})", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    print(f"Training finished. Best validation Dice: {best_dice:.4f}", flush=True)

    # Threshold sweep on the best checkpoint. Runs the model forward ONCE per validation
    # image (not once per threshold -- the forward pass doesn't depend on the threshold at
    # all) under autocast, matching the training loop; the earlier version of this section
    # did neither and took ~15-20x longer than the entire rest of training combined.
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict({k: v.float() for k, v in ckpt["model_state_dict"].items()})
    model.eval()
    all_probs, all_masks = [], []
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
            all_probs.append(torch.sigmoid(logits).float().cpu())
            all_masks.append(masks.cpu())
    all_probs = torch.cat(all_probs)
    all_masks = torch.cat(all_masks)

    best_thresh, best_thresh_dice = 0.5, 0.0
    for thresh in [0.24, 0.28, 0.32, 0.36, 0.40, 0.44, 0.50]:
        preds = (all_probs > thresh).float()
        inter = (preds * all_masks).sum().item()
        avg = 2 * inter / (preds.sum().item() + all_masks.sum().item() + 1e-6)
        print(f"Threshold {thresh:.2f} -> Dice {avg:.4f}", flush=True)
        if avg > best_thresh_dice:
            best_thresh_dice, best_thresh = avg, thresh
    print(f"Peak operational threshold: {best_thresh} (Dice {best_thresh_dice:.4f})", flush=True)

    with open(os.path.join(ROOT, "experiments", "mask2former_v3_config.json"), "w") as f:
        json.dump({"image_size": args.image_size, "num_queries": args.num_queries, "disk_shrink": DISK_SHRINK,
                   "epochs_run": len(history), "best_val_dice": best_dice,
                   "best_threshold": best_thresh, "best_threshold_dice": best_thresh_dice}, f, indent=2)


if __name__ == "__main__":
    main()
