"""
Solar Filament Super-Resolution Training Pipeline
==================================================
Trains domain-specific lightweight SR models on solar filament crops
using composite Charbonnier + SSIM loss.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from experiments.super_resolution.models import (
    SolarSRNet, EDSRSmall, ESPCN, FSRCNN, CompositeSRLoss, CharbonnierLoss, SSIMLoss
)
from experiments.super_resolution.training.dataset import SolarFilamentSRDataset


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) in dB."""
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return 100.0
    return 10.0 * np.log10(1.0 / mse)


def compute_ssim_val(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute SSIM metric value (1.0 = identical)."""
    loss_module = SSIMLoss(channels=pred.size(1)).to(pred.device)
    loss = loss_module(pred, target).item()
    return max(0.0, 1.0 - loss)


def train_sr_model(
    scale_factor: int = 2,
    model_name: str = "solar_sr",
    epochs: int = 25,
    batch_size: int = 16,
    lr: float = 5e-4,
    patch_size: int = 96,
    output_dir: str = "experiments/super_resolution/results",
    device_str: str = "auto"
):
    """
    Train a lightweight Super-Resolution model on solar filament patches.
    """
    os.makedirs(output_dir, exist_ok=True)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    image_dir = os.path.join(project_root, "assets/gallery_samples/images")
    mask_dir = os.path.join(project_root, "assets/gallery_samples/masks")

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"==================================================")
    print(f"Training SR Model: {model_name.upper()} ({scale_factor}x)")
    print(f"Device: {device}")
    if torch.cuda.is_available() and device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f"==================================================")

    # 1. Dataset & DataLoaders
    full_dataset = SolarFilamentSRDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        patch_size=patch_size,
        scale_factor=scale_factor,
        num_patches_per_image=25,
        augment=True,
        seed=42
    )

    total_samples = len(full_dataset)
    val_size = max(10, int(0.15 * total_samples))
    train_size = total_samples - val_size

    train_set, val_set = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    print(f"Dataset: {total_samples} total patches ({train_size} train, {val_size} val)")

    # 2. Instantiate Model
    if model_name == "solar_sr":
        model = SolarSRNet(scale_factor=scale_factor, in_channels=1, num_features=48, num_blocks=4)
    elif model_name == "edsr_small":
        model = EDSRSmall(scale_factor=scale_factor, in_channels=1, num_features=64, num_blocks=8)
    elif model_name == "espcn":
        model = ESPCN(scale_factor=scale_factor, in_channels=1, hidden_channels=64)
    elif model_name == "fsrcnn":
        model = FSRCNN(scale_factor=scale_factor, in_channels=1)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total_params:,} ({total_params / 1e3:.1f}K)")

    # 3. Loss & Optimizer
    criterion = CompositeSRLoss(alpha=0.8, beta=0.2, channels=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 4. Training Loop
    history = []
    best_val_psnr = -1.0
    best_checkpoint_path = os.path.join(output_dir, f"best_sr_model_{model_name}_x{scale_factor}.pt")
    default_best_path = os.path.join(output_dir, f"best_sr_model_x{scale_factor}.pt")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_psnr = 0.0
        n_train_batches = 0

        for lr_batch, hr_batch in train_loader:
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)

            optimizer.zero_grad()
            sr_batch = model(lr_batch)
            loss = criterion(sr_batch, hr_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_psnr += compute_psnr(sr_batch.detach(), hr_batch)
            n_train_batches += 1

        scheduler.step()
        train_loss /= max(1, n_train_batches)
        train_psnr /= max(1, n_train_batches)

        # Validation
        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        val_ssim = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for lr_batch, hr_batch in val_loader:
                lr_batch = lr_batch.to(device)
                hr_batch = hr_batch.to(device)
                sr_batch = model(lr_batch)
                loss = criterion(sr_batch, hr_batch)

                val_loss += loss.item()
                val_psnr += compute_psnr(sr_batch, hr_batch)
                val_ssim += compute_ssim_val(sr_batch, hr_batch)
                n_val_batches += 1

        val_loss /= max(1, n_val_batches)
        val_psnr /= max(1, n_val_batches)
        val_ssim /= max(1, n_val_batches)

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_psnr": round(train_psnr, 2),
            "val_loss": round(val_loss, 5),
            "val_psnr": round(val_psnr, 2),
            "val_ssim": round(val_ssim, 4),
            "lr": round(scheduler.get_last_lr()[0], 7)
        }
        history.append(epoch_record)

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} PSNR: {train_psnr:.2f}dB | "
              f"Val Loss: {val_loss:.4f} PSNR: {val_psnr:.2f}dB SSIM: {val_ssim:.4f}")

        # Save best model
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            checkpoint_data = {
                "model_name": model_name,
                "scale_factor": scale_factor,
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_psnr": val_psnr,
                "val_ssim": val_ssim,
                "config": {
                    "scale_factor": scale_factor,
                    "in_channels": 1,
                    "model_name": model_name
                }
            }
            torch.save(checkpoint_data, best_checkpoint_path)
            if model_name == "solar_sr":
                torch.save(checkpoint_data, default_best_path)

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.1f}s. Best Val PSNR: {best_val_psnr:.2f} dB")

    # Save artifacts
    history_df = pd.DataFrame(history)
    history_path = os.path.join(output_dir, f"training_history_{model_name}_x{scale_factor}.csv")
    history_df.to_csv(history_path, index=False)
    if model_name == "solar_sr":
        history_df.to_csv(os.path.join(output_dir, f"training_history_x{scale_factor}.csv"), index=False)

    config_info = {
        "model_name": model_name,
        "scale_factor": scale_factor,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "patch_size": patch_size,
        "total_params": total_params,
        "best_val_psnr": best_val_psnr,
        "best_val_ssim": history_df["val_ssim"].iloc[history_df["val_psnr"].idxmax()],
        "training_time_seconds": round(elapsed, 2)
    }
    with open(os.path.join(output_dir, f"config_{model_name}_x{scale_factor}.json"), "w") as f:
        json.dump(config_info, f, indent=2)

    return config_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Solar Filament SR")
    parser.add_argument("--scale", type=int, default=2, choices=[2, 4])
    parser.add_argument("--model", type=str, default="solar_sr", choices=["solar_sr", "edsr_small", "espcn", "fsrcnn"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    train_sr_model(
        scale_factor=args.scale,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
