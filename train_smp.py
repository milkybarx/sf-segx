"""
Trains a solar filament segmentation model on the MAGFiLO ground-truth masks
derived from the Kaggle "filament-segmentation-2026" competition data. Mirrors
the pipeline in Copy_of_FINAL_SOLAR.ipynb, with the Frangi-filter pseudo-labels
replaced by the real annotated masks produced by prepare_masks.py.

Supports multiple interchangeable architectures (same training loop, just a
different model instantiation), selected via --arch:

    python train_smp.py --arch unet_resnet34
    python train_smp.py --arch deeplabv3plus_resnet50 --epochs 35
"""
import argparse
import glob
import os
import time

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import uniform_filter
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import segmentation_models_pytorch as smp

DATA_ROOT = "data/MAGFiLO_1.0_Kaggle_2026/train"
IMG_DIR = os.path.join(DATA_ROOT, "train_images")
MASK_DIR = os.path.join(DATA_ROOT, "train_masks")
OUT_DIR = "outputs"          # logs, training summaries
CHECKPOINT_DIR = "checkpoints"  # model weights (shared with training/train.py's checkpoints)
# kept for backwards compatibility with existing imports (webapp/app.py smoke tests, etc.)
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "unet_resnet34_best.pth")

SEED = 42
IMG_SIZE = 512
BATCH_SIZE = 8
EPOCHS = 15
NUM_WORKERS = 0  # Windows uses spawn for multiprocessing, which can't pickle cv2.CLAHE


MODEL_REGISTRY = {
    "unet_resnet34": {
        "label": "U-Net (ResNet-34)",
        "build": lambda: smp.Unet(
            encoder_name="resnet34", encoder_weights="imagenet",
            in_channels=1, classes=1, activation=None,
        ),
        "checkpoint": "unet_resnet34_best.pth",
    },
    "deeplabv3plus_resnet50": {
        # ResNet-50 encoder, not EfficientNet-B4 as originally scoped: EfficientNet-B4's
        # depthwise-separable convs measured ~7-12x slower than plain ResNet convs on this
        # GPU/driver/cuDNN combo (3.7s/it vs 0.48s/it at batch=8, 512x512) and even hit a
        # CUDA OOM before AMP was added. One backbone per architecture -- this is DeepLabV3+'s.
        "label": "DeepLabV3+ (ResNet-50)",
        "build": lambda: smp.DeepLabV3Plus(
            encoder_name="resnet50", encoder_weights="imagenet",
            in_channels=1, classes=1, activation=None,
        ),
        "checkpoint": "deeplabv3plus_resnet50_best.pth",
    },
    "attention_unet": {
        # MONAI's AttentionUnet is a from-scratch architecture (no swappable ImageNet
        # encoder backbone, unlike the smp models above) -- attention gates on the skip
        # connections instead of a pretrained encoder.
        "label": "Attention U-Net (MONAI)",
        "build": lambda: __import__("monai.networks.nets", fromlist=["AttentionUnet"]).AttentionUnet(
            spatial_dims=2, in_channels=1, out_channels=1,
            channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2),
        ),
        "checkpoint": "attention_unet_best.pth",
    },
}


class GONGPreprocessor:
    def __init__(self, clip_limit: float = 2.5, grid_size: tuple = (8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)

    def find_solar_disk(self, gray_img: np.ndarray):
        h, w = gray_img.shape
        _, binary = cv2.threshold(gray_img, 25, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_cnt = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(largest_cnt)
            return int(x), int(y), int(radius * 0.97)
        return w // 2, h // 2, int(min(h, w) * 0.44)

    def preprocess(self, image: np.ndarray):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        h, w = gray.shape
        cx, cy, radius = self.find_solar_disk(gray)

        y_grid, x_grid = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
        disk_mask = dist_from_center <= radius

        valid_pixels = gray[disk_mask]
        if len(valid_pixels) > 0:
            p1, p99 = np.percentile(valid_pixels, (1.0, 99.0))
            clipped = np.clip(gray, p1, p99)
            norm = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            norm = gray

        bg = uniform_filter(norm.astype(np.float32), size=h // 8)
        bg[bg == 0] = 1.0
        flattened = norm.astype(np.float32) / bg
        scale_val = norm[disk_mask].max() / (bg[disk_mask].max() + 1e-6)
        flattened_disk = np.clip(flattened * 255.0 / (scale_val + 1e-6), 0, 255).astype(np.uint8)

        enhanced = self.clahe.apply(flattened_disk)
        enhanced[~disk_mask] = 0

        return enhanced, disk_mask


class SolarFilamentDataset(Dataset):
    def __init__(self, image_paths, mask_dir, preprocessor, img_size=512, is_train=True):
        self.image_paths = image_paths
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.preprocessor = preprocessor

        if is_train:
            self.transforms = A.Compose([
                A.Resize(img_size, img_size),
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Normalize(mean=(0.5,), std=(0.5,)),
                ToTensorV2()
            ])
        else:
            self.transforms = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=(0.5,), std=(0.5,)),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def _robust_imread(path, flag, retries=5, delay=0.1):
        """cv2.imread transiently returns None under concurrent access on Windows (e.g.
        a cloud-synced Downloads folder momentarily locking a file another process/dashboard
        just read) -- retry with a short backoff instead of crashing the whole run."""
        for attempt in range(retries):
            img = cv2.imread(path, flag)
            if img is not None:
                return img
            time.sleep(delay * (attempt + 1))
        raise FileNotFoundError(f"Could not read {path} after {retries} attempts (file may be missing/corrupt)")

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        fn = os.path.splitext(os.path.basename(img_path))[0]

        cache_dir = os.path.join(DATA_ROOT, "train_preprocessed")
        enh_path = os.path.join(cache_dir, fn + "_enh.png")
        disk_path = os.path.join(cache_dir, fn + "_disk.png")
        if os.path.exists(enh_path) and os.path.exists(disk_path):
            enhanced_img = self._robust_imread(enh_path, cv2.IMREAD_GRAYSCALE)
            disk_mask = self._robust_imread(disk_path, cv2.IMREAD_GRAYSCALE) > 127
        else:
            raw_img = self._robust_imread(img_path, cv2.IMREAD_GRAYSCALE)
            enhanced_img, disk_mask = self.preprocessor.preprocess(raw_img)

        mask_path = os.path.join(self.mask_dir, fn + ".png")
        gt_mask = self._robust_imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = (gt_mask > 127).astype(np.float32)
        gt_mask[~disk_mask] = 0.0

        augmented = self.transforms(image=enhanced_img, mask=gt_mask)
        return augmented['image'], augmented['mask'].unsqueeze(0)


class CompoundSolarLoss(nn.Module):
    def __init__(self, alpha: float = 0.8, gamma: float = 2.0, smooth: float = 1e-5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal = (self.alpha * (1 - p_t) ** self.gamma * bce).mean()

        probs_f = probs.view(-1)
        targets_f = targets.view(-1)
        intersection = (probs_f * targets_f).sum()
        dice = 1.0 - (2.0 * intersection + self.smooth) / (probs_f.sum() + targets_f.sum() + self.smooth)

        return focal + dice


def calculate_metrics(logits, targets, threshold=0.5):
    preds = (torch.sigmoid(logits) > threshold).float()
    intersection = (preds * targets).sum().item()
    total_union = (preds + targets).clamp(0, 1).sum().item()
    dice = (2.0 * intersection) / (preds.sum().item() + targets.sum().item() + 1e-6)
    iou = intersection / (total_union + 1e-6)
    return dice, iou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=list(MODEL_REGISTRY.keys()), default="unet_resnet34")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    spec = MODEL_REGISTRY[args.arch]
    checkpoint_path = os.path.join(CHECKPOINT_DIR, spec["checkpoint"])
    epochs = args.epochs
    batch_size = args.batch_size

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Environment initialized on: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Architecture: {spec['label']} ({args.arch})")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    preprocessor = GONGPreprocessor()

    image_paths = sorted(
        glob.glob(f"{IMG_DIR}/*.jpg") +
        glob.glob(f"{IMG_DIR}/*.jpeg") +
        glob.glob(f"{IMG_DIR}/*.png")
    )
    print(f"Total training images: {len(image_paths)}")
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found in {IMG_DIR}")

    full_dataset = SolarFilamentDataset(image_paths, MASK_DIR, preprocessor, img_size=IMG_SIZE, is_train=True)
    val_size = max(2, int(0.15 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )
    # validation split must use eval-time transforms (no augmentation)
    val_ds.dataset = SolarFilamentDataset(image_paths, MASK_DIR, preprocessor, img_size=IMG_SIZE, is_train=False)

    # drop_last: a size-1 tail batch crashes BatchNorm in decoders with a global-pooling
    # branch (e.g. DeepLabV3+'s ASPP), since a 1x1xN feature map has only 1 value/channel.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    print(f"DataLoaders ready: {train_size} training images | {val_size} validation images (batch_size={batch_size})")

    model = spec["build"]().to(device)

    criterion = CompoundSolarLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    print(f"Model built: {spec['label']}.")

    best_val_dice = 0.0
    print(f"Training {spec['label']} for {epochs} epochs on {device}...")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_dice = 0.0, 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs} [train]", leave=False)
        for imgs, masks in train_bar:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(imgs)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            d, _ = calculate_metrics(logits, masks)
            train_loss += loss.item()
            train_dice += d
            train_bar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{d:.4f}")

        train_loss /= len(train_loader)
        train_dice /= len(train_loader)

        model.eval()
        val_loss, val_dice, val_iou = 0.0, 0.0, 0.0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch:02d}/{epochs} [val]", leave=False)
        with torch.no_grad():
            for imgs, masks in val_bar:
                imgs, masks = imgs.to(device), masks.to(device)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    logits = model(imgs)
                    loss = criterion(logits, masks)

                d, iou = calculate_metrics(logits, masks)
                val_loss += loss.item()
                val_dice += d
                val_iou += iou
                val_bar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{d:.4f}")

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_iou /= len(val_loader)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"| Train Loss: {train_loss:.4f} - Dice: {train_dice:.4f} "
              f"| Val Loss: {val_loss:.4f} - Dice: {val_dice:.4f} - IoU: {val_iou:.4f}", flush=True)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), checkpoint_path)
            print(f"   Saved best checkpoint (Dice: {val_dice:.4f})", flush=True)

    print(f"Training finished. Best validation Dice: {best_val_dice:.4f}")

    # Threshold optimization on the best checkpoint
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    best_thresh, highest_dice = 0.5, 0.0
    thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
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
        print(f"Threshold: {thresh:.2f} -> Validation Dice: {avg_dice:.4f}", flush=True)
        if avg_dice > highest_dice:
            highest_dice = avg_dice
            best_thresh = thresh

    print(f"Optimal threshold: {best_thresh} (Dice: {highest_dice:.4f})")

    with open(os.path.join(OUT_DIR, f"training_summary_{args.arch}.txt"), "w") as f:
        f.write(f"arch={args.arch}\n")
        f.write(f"label={spec['label']}\n")
        f.write(f"checkpoint={checkpoint_path}\n")
        f.write(f"best_val_dice={best_val_dice:.4f}\n")
        f.write(f"best_threshold={best_thresh}\n")
        f.write(f"best_threshold_dice={highest_dice:.4f}\n")
        f.write(f"epochs={epochs}\n")
        f.write(f"train_size={train_size}\n")
        f.write(f"val_size={val_size}\n")


if __name__ == "__main__":
    main()
