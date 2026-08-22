"""
Training Script
===============
Train U-Net for solar filament segmentation on NVIDIA RTX 4050.
Supports Automatic Mixed Precision (AMP), gradient scaling, checkpointing,
and comprehensive metric tracking.
"""

import os
import sys
import time
import json
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from typing import Dict, Tuple
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet import build_unet
from training.losses import DiceBCELoss
from training.metrics import compute_all_metrics
from preprocessing.dataset import get_dataloaders


def setup_device(config: dict) -> torch.device:
    """Setup and verify CUDA device."""
    print("=" * 60)
    print("DEVICE CONFIGURATION")
    print("=" * 60)

    if torch.cuda.is_available():
        gpu_idx = config.get('device', {}).get('gpu_index', 0)
        device = torch.device(f'cuda:{gpu_idx}')
        gpu_name = torch.cuda.get_device_name(gpu_idx)
        gpu_mem = torch.cuda.get_device_properties(gpu_idx).total_memory / (1024 ** 3)
        print(f"  GPU:         {gpu_name}")
        print(f"  VRAM:        {gpu_mem:.1f} GB")
        print(f"  CUDA:        {torch.version.cuda}")
        print(f"  PyTorch:     {torch.__version__}")
        print(f"  Device:      {device}")

        # Clear GPU cache
        torch.cuda.empty_cache()
    else:
        print("  WARNING: CUDA is NOT available!")
        print("  Training will use CPU (significantly slower)")
        device = torch.device('cpu')

    print("=" * 60)
    return device


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool,
    gradient_clip: float = 1.0,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()

        # Metrics (detach for speed)
        with torch.no_grad():
            metrics = compute_all_metrics(logits.detach(), masks)

        total_loss += loss.item()
        total_dice += metrics['dice']
        total_iou += metrics['iou']
        n_batches += 1

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'dice': f"{metrics['dice']:.4f}",
        })

    return {
        'loss': total_loss / max(n_batches, 1),
        'dice': total_dice / max(n_batches, 1),
        'iou': total_iou / max(n_batches, 1),
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    n_batches = 0

    for images, masks in tqdm(loader, desc="Validating", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)

        metrics = compute_all_metrics(logits, masks)

        total_loss += loss.item()
        total_dice += metrics['dice']
        total_iou += metrics['iou']
        total_precision += metrics['precision']
        total_recall += metrics['recall']
        n_batches += 1

    n = max(n_batches, 1)
    return {
        'loss': total_loss / n,
        'dice': total_dice / n,
        'iou': total_iou / n,
        'precision': total_precision / n,
        'recall': total_recall / n,
    }


def train(config_path: str = None):
    """Main training function."""
    # Load config
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    'configs', 'default_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup device
    device = setup_device(config)

    # Paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_root = os.path.join(project_root, config['data']['dataset_root'])
    image_dir = os.path.join(dataset_root, config['data']['train_images_dir'])
    annotations_json = os.path.join(dataset_root, config['data']['annotations_file'])

    # Create output directories
    checkpoint_dir = os.path.join(project_root, config['output']['checkpoint_dir'])
    experiment_dir = os.path.join(project_root, config['output']['experiment_dir'])
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(experiment_dir, exist_ok=True)

    # Data
    print("\nLoading data...")
    train_loader, val_loader = get_dataloaders(
        image_dir=image_dir,
        annotations_json=annotations_json,
        image_size=config['data']['image_size'],
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        train_ratio=config['data']['train_ratio'],
        seed=config['data']['seed'],
        pin_memory=device.type == 'cuda',
    )

    # Model
    model_name = config.get('model', {}).get('name', 'mask2former').lower()
    print(f"\nBuilding model architecture: [{model_name.upper()}]...")
    if model_name == 'mask2former':
        from models.mask2former import build_mask2former
        model = build_mask2former(config.get('model', {}))
    else:
        from models.unet import build_unet
        model = build_unet(config.get('model', {}))
    model = model.to(device)

    # Loss, optimizer, scheduler
    criterion = DiceBCELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['training']['epochs'], eta_min=1e-7
    )

    # AMP setup
    use_amp = config['training'].get('amp', True) and device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)
    gradient_clip = config['training'].get('gradient_clip', 1.0)
    print(f"AMP: {'Enabled' if use_amp else 'Disabled'}")

    # Training state
    best_val_dice = 0.0
    patience_counter = 0
    patience = config['training'].get('patience', 10)
    history = []

    print(f"\nStarting training for {config['training']['epochs']} epochs...")
    print(f"Train: {len(train_loader.dataset)} images, Val: {len(val_loader.dataset)} images")
    print(f"Batch size: {config['training']['batch_size']}")
    print()

    start_time = time.time()

    for epoch in range(1, config['training']['epochs'] + 1):
        epoch_start = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler, use_amp, gradient_clip
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, use_amp)

        # Scheduler step
        scheduler.step()

        epoch_time = time.time() - epoch_start
        lr = optimizer.param_groups[0]['lr']

        # Log
        print(f"Epoch {epoch:3d}/{config['training']['epochs']} | "
              f"Train Loss: {train_metrics['loss']:.4f} Dice: {train_metrics['dice']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} Dice: {val_metrics['dice']:.4f} "
              f"IoU: {val_metrics['iou']:.4f} P: {val_metrics['precision']:.4f} "
              f"R: {val_metrics['recall']:.4f} | "
              f"LR: {lr:.2e} | {epoch_time:.1f}s")

        # Save history
        epoch_record = {
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
            'lr': lr,
            'time': epoch_time,
        }
        history.append(epoch_record)

        # Best model checkpoint
        if val_metrics['dice'] > best_val_dice:
            best_val_dice = val_metrics['dice']
            patience_counter = 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': best_val_dice,
                'val_metrics': val_metrics,
                'config': config,
            }
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_model.pth'))
            print(f"  * [SAVED] New best model saved (Val Dice: {best_val_dice:.4f})")
        else:
            patience_counter += 1

        # Save latest
        if epoch % 10 == 0 or epoch == config['training']['epochs']:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
            }, os.path.join(checkpoint_dir, 'latest_model.pth'))

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time / 60:.1f} minutes")
    print(f"Best validation Dice: {best_val_dice:.4f}")

    # Save training history
    experiment_result = {
        'config': config,
        'best_val_dice': best_val_dice,
        'total_epochs': len(history),
        'total_time_minutes': total_time / 60,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'history': history,
    }

    with open(os.path.join(experiment_dir, 'training_results.json'), 'w') as f:
        json.dump(experiment_result, f, indent=2)

    print(f"Results saved to {experiment_dir}/training_results.json")

    return model, history


if __name__ == '__main__':
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    train(config_path)
