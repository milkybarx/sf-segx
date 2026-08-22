"""
Trains SegFormer-B2 (nvidia/mit-b2, 2-class head) on the MAGFiLO dataset.

Adapted from notebooks/SegFormer_B2_Solar_Filament_Training.ipynb (the actual recipe the
currently-shipped checkpoints/segformer_b2_best.pt was trained with -- ET3RYX's own notebook,
hardcoded to a local path that doesn't exist on this machine). Same data split, model,
augmentation, and optimizer as that notebook; this pass additionally tries:
  - --image_size 768 (vs. the shipped checkpoint's 640) with an automatic OOM fallback to
    640, the same pattern the notebook itself used to fall back from 640 to 512.
  - --loss focal_dice as an alternative to the shipped dice_bce (both were already coded
    as options in the source notebook, just never run against each other).

Resilient: saves a checkpoint on every validation Dice improvement (not just once at the
end), and writes per-epoch history to a CSV model_hub.py can chart on the Overview page.
"""
import argparse
import gc
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

IMAGE_DIR = os.path.join(ROOT, "data", "MAGFiLO_1.0_Kaggle_2026", "train", "train_images")
MASK_DIR = os.path.join(ROOT, "data", "MAGFiLO_1.0_Kaggle_2026", "train", "train_masks")
CHECKPOINT_PATH = os.path.join(ROOT, "checkpoints", "segformer_b2_best.pt")
HISTORY_PATH = os.path.join(ROOT, "experiments", "segformer_b2_history.csv")
MODEL_NAME = "nvidia/mit-b2"
SEED = 42
BATCH_SIZE = 1
GRAD_ACCUM = 4
ENCODER_LR = 1e-5
DECODER_LR = 1e-4
WEIGHT_DECAY = 1e-4
THRESHOLD = 0.5


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_records():
    image_files = sorted(f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    records = []
    for fn in image_files:
        stem = os.path.splitext(fn)[0]
        mask_path = os.path.join(MASK_DIR, stem + ".png")
        if os.path.exists(mask_path):
            records.append({"name": stem, "image": os.path.join(IMAGE_DIR, fn), "mask": mask_path})
    assert len(records) == 707, f"Expected 707 matched pairs, got {len(records)}"
    return records


def split_records(records):
    shuffled = records.copy()
    random.Random(SEED).shuffle(shuffled)
    return shuffled[:565], shuffled[565:636], shuffled[636:707]


def augment_pair(image, mask):
    if random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    quarter_turns = random.randrange(4)
    if quarter_turns:
        image = image.rotate(90 * quarter_turns, expand=True)
        mask = mask.rotate(90 * quarter_turns, expand=True)
    angle = random.uniform(-8, 8)
    image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
    mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=0)
    if random.random() < 0.5:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
    if random.random() < 0.5:
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.9, 1.1))
    return image, mask


class SolarFilamentDataset(Dataset):
    def __init__(self, records, image_size, processor, training=False):
        self.records, self.image_size, self.processor, self.training = records, image_size, processor, training

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = Image.open(record["image"]).convert("RGB")
        mask = Image.open(record["mask"]).convert("L")
        if self.training:
            image, mask = augment_pair(image, mask)
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        encoded = self.processor(images=np.asarray(image), do_resize=False, return_tensors="pt")
        labels = torch.from_numpy((np.asarray(mask) > 127).astype(np.int64))
        return {"pixel_values": encoded["pixel_values"].squeeze(0), "labels": labels}


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)[:, 1]
        targets = targets.float()
        dims = (1, 2)
        intersection = (probs * targets).sum(dims)
        return (1 - ((2 * intersection + self.smooth) / (probs.sum(dims) + targets.sum(dims) + self.smooth))).mean()


def focal_loss(logits, targets, gamma=2.0):
    fg_logits = logits[:, 1]
    bce = F.binary_cross_entropy_with_logits(fg_logits, targets.float(), reduction="none")
    probs = torch.sigmoid(fg_logits)
    modulating = torch.where(targets.bool(), 1 - probs, probs) ** gamma
    return (modulating * bce).mean()


def confusion_counts(logits, targets):
    preds = torch.softmax(logits, dim=1)[:, 1] >= THRESHOLD
    targets = targets.bool()
    return ((preds & targets).sum().item(), (preds & ~targets).sum().item(), (~preds & targets).sum().item())


def metrics_from_counts(tp, fp, fn):
    return {
        "dice": 2 * tp / max(2 * tp + fp + fn, 1),
        "iou": tp / max(tp + fp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--image_size", type=int, default=768)
    ap.add_argument("--loss", choices=["dice_bce", "focal_dice"], default="focal_dice")
    ap.add_argument("--early_stopping_patience", type=int, default=8)
    args = ap.parse_args()

    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    records = build_records()
    train_records, val_records, test_records = split_records(records)
    print(f"Train: {len(train_records)}  Val: {len(val_records)}  Test: {len(test_records)}", flush=True)

    processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)

    def make_model():
        return SegformerForSemanticSegmentation.from_pretrained(
            MODEL_NAME, num_labels=2, id2label={0: "background", 1: "filament"},
            label2id={"background": 0, "filament": 1}, ignore_mismatched_sizes=True,
        ).to(device)

    def make_optimizer_scheduler(model):
        encoder, decoder = [], []
        for name, p in model.named_parameters():
            (decoder if "decode_head" in name else encoder).append(p)
        opt = torch.optim.AdamW(
            [{"params": encoder, "lr": ENCODER_LR}, {"params": decoder, "lr": DECODER_LR}],
            weight_decay=WEIGHT_DECAY,
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3, threshold=1e-3, min_lr=1e-7)
        return opt, sched

    def make_loaders(image_size):
        common = dict(batch_size=BATCH_SIZE, num_workers=0, pin_memory=(device.type == "cuda"))
        train_ds = SolarFilamentDataset(train_records, image_size, processor, True)
        val_ds = SolarFilamentDataset(val_records, image_size, processor, False)
        test_ds = SolarFilamentDataset(test_records, image_size, processor, False)
        return (train_ds, val_ds, test_ds,
                DataLoader(train_ds, shuffle=True, **common),
                DataLoader(val_ds, shuffle=False, **common),
                DataLoader(test_ds, shuffle=False, **common))

    image_size = args.image_size
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = make_loaders(image_size)
    model = make_model()
    optimizer, scheduler = make_optimizer_scheduler(model)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SegFormer-B2: {n_params:,} params, image_size={image_size}, loss={args.loss}", flush=True)

    def forward_logits(batch):
        outputs = model(pixel_values=batch["pixel_values"].to(device))
        return F.interpolate(outputs.logits, size=batch["labels"].shape[-2:], mode="bilinear", align_corners=False)

    def compute_loss(logits, targets):
        dice = DiceLoss()(logits, targets)
        if args.loss == "dice_bce":
            return 0.7 * dice + 0.3 * nn.functional.binary_cross_entropy_with_logits(logits[:, 1], targets.float())
        return 0.7 * dice + 0.3 * focal_loss(logits, targets)

    # One-batch VRAM test with automatic fallback (768 -> 640), same pattern the source
    # notebook used (640 -> 512) -- this machine's GPU is the same class (6GB) it was tuned for.
    try:
        model.train()
        batch = next(iter(train_loader))
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = forward_logits(batch)
            loss = compute_loss(logits, batch["labels"].to(device))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        print(f"One-batch test passed at image_size={image_size}", flush=True)
    except RuntimeError as e:
        if "out of memory" not in str(e).lower() or image_size == 640:
            raise
        print(f"OOM at {image_size} -- falling back to 640", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        image_size = 640
        train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = make_loaders(image_size)
        model = make_model()
        optimizer, scheduler = make_optimizer_scheduler(model)
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    def train_one_epoch():
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss, tp, fp, fn = 0.0, 0, 0, 0
        for step, batch in enumerate(train_loader):
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = forward_logits(batch)
                loss = compute_loss(logits, batch["labels"].to(device)) / GRAD_ACCUM
            scaler.scale(loss).backward()
            if (step + 1) % GRAD_ACCUM == 0 or step + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total_loss += loss.item() * GRAD_ACCUM
            a, b, c = confusion_counts(logits.detach(), batch["labels"].to(device))
            tp += a; fp += b; fn += c
        return total_loss / len(train_loader), metrics_from_counts(tp, fp, fn)

    @torch.no_grad()
    def validate():
        model.eval()
        total_loss, tp, fp, fn = 0.0, 0, 0, 0
        for batch in val_loader:
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = forward_logits(batch)
                total_loss += compute_loss(logits, batch["labels"].to(device)).item()
            a, b, c = confusion_counts(logits, batch["labels"].to(device))
            tp += a; fp += b; fn += c
        return total_loss / len(val_loader), metrics_from_counts(tp, fp, fn)

    def save_checkpoint(epoch, best_dice):
        torch.save({
            "epoch": epoch,
            "model_state_dict": {k: v.half() for k, v in model.state_dict().items()},
            "best_validation_dice": best_dice,
            "configuration": {"loss_type": args.loss, "batch_size": BATCH_SIZE, "grad_accumulation": GRAD_ACCUM},
            "image_size": image_size,
            "model_name": MODEL_NAME,
            "seed": SEED,
        }, CHECKPOINT_PATH)

    history = []
    best_dice = -float("inf")
    patience_counter = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_m = train_one_epoch()
        val_loss, val_m = validate()
        scheduler.step(val_m["dice"])
        lr = max(g["lr"] for g in optimizer.param_groups)
        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_dice": train_m["dice"], "train_iou": train_m["iou"],
            "val_loss": val_loss, "val_dice": val_m["dice"], "val_iou": val_m["iou"],
            "val_precision": val_m["precision"], "val_recall": val_m["recall"], "lr": lr,
        })
        pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)
        print(f"Epoch [{epoch}/{args.epochs}] | Train Loss: {train_loss:.4f} - Dice: {train_m['dice']:.4f} | "
              f"Val Loss: {val_loss:.4f} - Dice: {val_m['dice']:.4f} - IoU: {val_m['iou']:.4f} | "
              f"LR: {lr:.2e} | ({time.time() - t0:.1f}s)", flush=True)
        if val_m["dice"] > best_dice:
            best_dice = val_m["dice"]
            patience_counter = 0
            save_checkpoint(epoch, best_dice)
            print(f"   Saved best checkpoint (Dice: {best_dice:.4f})", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {args.early_stopping_patience} epochs)", flush=True)
                break

    print(f"Training finished. Best validation Dice: {best_dice:.4f}", flush=True)

    with open(os.path.join(ROOT, "experiments", "segformer_b2_config.json"), "w") as f:
        json.dump({
            "model_name": "segformer_b2", "image_size": image_size, "loss": args.loss,
            "epochs_run": len(history), "best_val_dice": best_dice,
        }, f, indent=2)


if __name__ == "__main__":
    main()
