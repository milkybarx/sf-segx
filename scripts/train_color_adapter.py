"""
Trains models/color_adapter.py's ColorToHAlphaNet.

No real color solar imagery exists anywhere in this repo/dataset (every MAGFiLO training
JPEG is confirmed true grayscale, R==G==B exactly) -- so there's no paired (color, H-alpha)
data to learn from directly. Instead this trains self-supervised: take a real grayscale
H-alpha image, synthetically re-color it with a random hue/saturation/gamma transform, and
train the network to recover the original grayscale from the synthetic color version. A wide
enough spread of random tints forces the network to learn a general "undo an unknown global
color cast" mapping instead of memorizing one look.

Usage: python scripts/train_color_adapter.py [--epochs 15] [--batch_size 16]
"""
import argparse
import glob
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.color_adapter import build_color_adapter
IMG_DIR = os.path.join(ROOT, "data", "MAGFiLO_1.0_Kaggle_2026", "train", "train_images")
CROP = 320


def synth_colorize(gray_u8: np.ndarray, rng: random.Random) -> np.ndarray:
    """gray_u8: (H,W) uint8 -> synthetic colored (H,W,3) uint8 BGR."""
    if rng.random() < 0.15:
        # Leave a slice of examples truly grayscale so the network also learns
        # a clean identity mapping for input that's already correct.
        return cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)

    gamma = rng.uniform(0.7, 1.4)
    v = np.clip((gray_u8.astype(np.float32) / 255.0) ** gamma * 255.0, 0, 255).astype(np.uint8)
    h = np.full_like(v, rng.randint(0, 179), dtype=np.uint8)
    s = np.full_like(v, int(rng.uniform(0.15, 0.95) * 255), dtype=np.uint8)
    hsv = np.stack([h, s, v], axis=-1)
    colored = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Small extra per-channel gain jitter so it's not *purely* a hue rotation.
    gains = np.array([rng.uniform(0.9, 1.1) for _ in range(3)], dtype=np.float32)
    colored = np.clip(colored.astype(np.float32) * gains, 0, 255).astype(np.uint8)
    return colored


class SyntheticColorDataset(Dataset):
    """images: pre-decoded grayscale arrays held in RAM (see main()) -- re-reading/decoding
    707 full-res JPEGs from disk every single epoch (num_workers=0) was the actual
    bottleneck in an earlier run of this script, not GPU compute; decoding once up front
    and reusing the arrays in memory for every epoch avoids that entirely."""

    def __init__(self, images, crop: int = CROP, seed: int = 0):
        self.images = images
        self.crop = crop
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        h, w = img.shape
        c = self.crop
        y = self.rng.randint(0, max(h - c, 0))
        x = self.rng.randint(0, max(w - c, 0))
        crop = img[y:y + c, x:x + c]
        if crop.shape != (c, c):
            crop = cv2.resize(crop, (c, c))

        if self.rng.random() < 0.5:
            crop = np.ascontiguousarray(crop[:, ::-1])
        if self.rng.random() < 0.5:
            crop = np.ascontiguousarray(crop[::-1, :])

        colored = synth_colorize(crop, self.rng)
        colored_t = torch.from_numpy(colored[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
        target_t = torch.from_numpy(crop.copy()).unsqueeze(0).float() / 255.0
        return colored_t, target_t


_SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
_SOBEL_Y = _SOBEL_X.transpose(2, 3)


def gradient_l1(pred, target, device):
    sx, sy = _SOBEL_X.to(device), _SOBEL_Y.to(device)
    pgx, pgy = nn.functional.conv2d(pred, sx, padding=1), nn.functional.conv2d(pred, sy, padding=1)
    tgx, tgy = nn.functional.conv2d(target, sx, padding=1), nn.functional.conv2d(target, sy, padding=1)
    return (pgx - tgx).abs().mean() + (pgy - tgy).abs().mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    paths = sorted(glob.glob(f"{IMG_DIR}/*.jpeg"))
    assert paths, f"No training images found under {IMG_DIR}"
    val_paths, train_paths = paths[:50], paths[50:]

    print(f"Decoding {len(paths)} images into memory once (avoids re-reading from disk "
          f"every epoch)...", flush=True)
    t0 = time.time()
    train_imgs = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in train_paths]
    val_imgs = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in val_paths]
    print(f"Done in {time.time() - t0:.1f}s. Train images: {len(train_imgs)}  "
          f"Val images: {len(val_imgs)}", flush=True)

    train_ds = SyntheticColorDataset(train_imgs, seed=0)
    val_ds = SyntheticColorDataset(val_imgs, seed=123)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_color_adapter(base=24).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ColorToHAlphaNet: {n_params:,} params", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    out_path = os.path.join(ROOT, "checkpoints", "color_to_halpha_adapter.pth")
    best_val = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        model.train()
        train_loss = 0.0
        for colored, target in train_dl:
            colored, target = colored.to(device), target.to(device)
            opt.zero_grad()
            pred = model(colored)
            loss = nn.functional.l1_loss(pred, target) + 0.15 * gradient_l1(pred, target, device)
            loss.backward()
            opt.step()
            train_loss += loss.item() * colored.size(0)
        train_loss /= len(train_ds)
        sched.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for colored, target in val_dl:
                colored, target = colored.to(device), target.to(device)
                pred = model(colored)
                val_loss += nn.functional.l1_loss(pred, target).item() * colored.size(0)
        val_loss /= len(val_ds)

        print(f"Epoch [{epoch}/{args.epochs}] Train Loss: {train_loss:.4f}  Val L1: {val_loss:.4f}  "
              f"({time.time() - t_epoch:.1f}s)", flush=True)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().half() for k, v in model.state_dict().items()}
            # Saved on every improvement (not just once at the end) so an interruption
            # mid-run never loses all progress.
            torch.save({
                "model_state_dict": best_state,
                "epoch": epoch,
                "val_l1": best_val,
                "config": {"base": 24, "crop": CROP},
            }, out_path)

    print(f"Saved best checkpoint (val_l1={best_val:.4f}) -> {out_path}")


if __name__ == "__main__":
    main()
