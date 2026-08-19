"""
Finalizes the attention_unet checkpoint that was saved (epoch 20, Dice 0.6507) before the
training run crashed at epoch 22 on a transient file-read error (fixed in train_smp.py).
Runs the same threshold sweep + summary/log writing train_smp.py does at the end of a run,
without re-training (loss had already plateaued/was oscillating in the 0.63-0.65 range for
the last ~10 epochs, so a full re-run wasn't worth the ~35 extra minutes for the expected gain).
"""
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from train_smp import (
    GONGPreprocessor, SolarFilamentDataset, MODEL_REGISTRY, CHECKPOINT_DIR,
    IMG_DIR, MASK_DIR, IMG_SIZE, BATCH_SIZE, NUM_WORKERS, SEED,
)
import glob

ARCH = "attention_unet"
LOG_PATH = "outputs/logs/train_log_attention_unet.txt"


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preprocessor = GONGPreprocessor()
    image_paths = sorted(
        glob.glob(f"{IMG_DIR}/*.jpg") + glob.glob(f"{IMG_DIR}/*.jpeg") + glob.glob(f"{IMG_DIR}/*.png")
    )
    full_dataset = SolarFilamentDataset(image_paths, MASK_DIR, preprocessor, img_size=IMG_SIZE, is_train=True)
    val_size = max(2, int(0.15 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED)
    )
    val_ds.dataset = SolarFilamentDataset(image_paths, MASK_DIR, preprocessor, img_size=IMG_SIZE, is_train=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    spec = MODEL_REGISTRY[ARCH]
    checkpoint_path = os.path.join(CHECKPOINT_DIR, spec["checkpoint"])
    model = spec["build"]().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    best_thresh, highest_dice = 0.5, 0.0
    thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    lines = []
    for thresh in thresholds:
        total_dice = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                probs = torch.sigmoid(logits)
                preds = (probs > thresh).float()
                inter = (preds * masks).sum().item()
                dice = (2.0 * inter) / (preds.sum().item() + masks.sum().item() + 1e-6)
                total_dice += dice
        avg_dice = total_dice / len(val_loader)
        line = f"Threshold: {thresh:.2f} -> Validation Dice: {avg_dice:.4f}"
        print(line)
        lines.append(line)
        if avg_dice > highest_dice:
            highest_dice = avg_dice
            best_thresh = thresh

    final_line = f"Optimal threshold: {best_thresh} (Dice: {highest_dice:.4f})"
    print(final_line)
    lines.append(final_line)

    with open(LOG_PATH, "a") as f:
        f.write("\n" + "\n".join(lines) + "\n")

    with open(f"outputs/training_summary_{ARCH}.txt", "w") as f:
        f.write(f"arch={ARCH}\n")
        f.write(f"label={spec['label']}\n")
        f.write(f"checkpoint={checkpoint_path}\n")
        f.write(f"best_val_dice=0.6507 (epoch 20; run crashed epoch 22/25 on a transient "
                f"file-read error, fixed in train_smp.py; not re-run given plateaued trend)\n")
        f.write(f"best_threshold={best_thresh}\n")
        f.write(f"best_threshold_dice={highest_dice:.4f}\n")
        f.write(f"epochs=20 (of 25 planned)\n")
        f.write(f"train_size={train_size}\n")
        f.write(f"val_size={val_size}\n")


if __name__ == "__main__":
    main()
