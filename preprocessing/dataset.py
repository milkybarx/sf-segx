"""
Solar Filament Dataset with Fast Caching
=========================================
PyTorch Dataset class for the MAGFiLO solar filament dataset.
Handles COCO-format polygon annotations, pre-rendered 512x512 mask caching,
data augmentation, and GPU-optimized DataLoaders.
"""

import os
import json
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List, Dict
from preprocessing.solar_preprocessor import SolarPreprocessor


def coco_poly_to_mask(segmentation: List[List[float]], height: int, width: int) -> np.ndarray:
    """Convert COCO polygon segmentation to binary mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in segmentation:
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
        pts = pts.astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def load_coco_annotations(json_path: str) -> Tuple[Dict, Dict, Dict]:
    """Load COCO annotations and organize by image."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    images_dict = {img['id']: img for img in data['images']}
    categories = {cat['id']: cat for cat in data['categories']}

    annotations_by_image = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    return images_dict, annotations_by_image, categories


class SolarFilamentDataset(Dataset):
    """
    GPU-optimized PyTorch dataset with preprocessed 512x512 caching.
    """

    def __init__(
        self,
        image_dir: str,
        annotations_json: str,
        image_size: int = 512,
        augment: bool = False,
        image_ids: Optional[List[str]] = None,
        cache_dir: Optional[str] = "cache_512",
    ):
        self.image_dir = image_dir
        self.image_size = image_size
        self.augment = augment
        self.cache_dir = cache_dir
        self.preprocessor = SolarPreprocessor(target_size=image_size)

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        # Load annotations
        self.images_dict, self.annotations_by_image, self.categories = \
            load_coco_annotations(annotations_json)

        available_files = set(os.listdir(image_dir))

        if image_ids is not None:
            self.image_ids = [
                iid for iid in image_ids
                if iid in self.images_dict
                and self.images_dict[iid]['file_name'] in available_files
                and iid in self.annotations_by_image
            ]
        else:
            self.image_ids = [
                iid for iid, img in self.images_dict.items()
                if img['file_name'] in available_files
                and iid in self.annotations_by_image
            ]

    def __len__(self):
        return len(self.image_ids)

    def _generate_mask(self, image_id: str, height: int, width: int) -> np.ndarray:
        """Generate binary segmentation mask from COCO polygon annotations."""
        mask = np.zeros((height, width), dtype=np.uint8)
        annotations = self.annotations_by_image.get(image_id, [])
        for ann in annotations:
            seg = ann.get('segmentation', [])
            if seg:
                ann_mask = coco_poly_to_mask(seg, height, width)
                mask = np.maximum(mask, ann_mask)
        return mask

    def _augment(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply fast on-the-fly augmentation."""
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        if random.random() > 0.5:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()

        k = random.randint(0, 3)
        if k > 0:
            image = np.rot90(image, k).copy()
            mask = np.rot90(mask, k).copy()

        if random.random() > 0.5:
            factor = random.uniform(0.85, 1.15)
            image = np.clip(image * factor, 0.0, 1.0).astype(np.float32)

        return image, mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_id = self.image_ids[idx]
        img_info = self.images_dict[image_id]
        file_name = img_info['file_name']
        base_id = os.path.splitext(file_name)[0]
        cache_dir = getattr(self, 'cache_dir', 'cache_512')
        cached_img_path = os.path.join(cache_dir, f"{base_id}_img.npy") if cache_dir else None
        cached_mask_path = os.path.join(cache_dir, f"{base_id}_mask.npy") if cache_dir else None

        if cached_img_path and os.path.exists(cached_img_path) and os.path.exists(cached_mask_path):
            # Ultra-fast load from cache (<1 ms)
            preprocessed = np.load(cached_img_path)
            mask_float = np.load(cached_mask_path)
        else:
            # First-time process & cache
            img_path = os.path.join(self.image_dir, file_name)
            raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if raw is None:
                raise FileNotFoundError(f"Could not load: {img_path}")

            orig_h, orig_w = raw.shape[:2]
            raw_mask = self._generate_mask(image_id, orig_h, orig_w)

            # Preprocess image
            preprocessed = self.preprocessor.preprocess_for_model(raw)

            # Resize mask
            mask_resized = cv2.resize(raw_mask, (self.image_size, self.image_size),
                                      interpolation=cv2.INTER_NEAREST)
            mask_float = mask_resized.astype(np.float32)

            # Save to cache
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                np.save(cached_img_path, preprocessed)
                np.save(cached_mask_path, mask_float)

        if self.augment:
            preprocessed, mask_float = self._augment(preprocessed, mask_float)

        image_tensor = torch.from_numpy(preprocessed).unsqueeze(0)  # [1, H, W]
        mask_tensor = torch.from_numpy(mask_float).unsqueeze(0)     # [1, H, W]

        return image_tensor, mask_tensor


def create_data_splits(
    annotations_json: str,
    image_dir: str,
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Create reproducible train/validation splits.

    Splits by unique file_name, not by raw COCO image_id: the same physical image can
    appear under multiple image_id entries (multiple annotation sessions -- measured 296
    of 707 files in MAGFiLO have 2-3 duplicate sessions). Splitting by image_id lets 121
    of those files leak across train/val (same picture seen in both, just a different
    session's mask), inflating validation Dice. All ids for a given file now move
    together into either train or val.
    """
    images_dict, annotations_by_image, _ = load_coco_annotations(annotations_json)
    available_files = set(os.listdir(image_dir))

    valid_ids = [
        iid for iid, img in images_dict.items()
        if img['file_name'] in available_files
        and iid in annotations_by_image
    ]

    ids_by_filename: Dict[str, List[str]] = {}
    for iid in valid_ids:
        ids_by_filename.setdefault(images_dict[iid]['file_name'], []).append(iid)

    filenames = sorted(ids_by_filename.keys())
    rng = random.Random(seed)
    rng.shuffle(filenames)

    split_idx = int(len(filenames) * train_ratio)
    train_files = filenames[:split_idx]
    val_files = filenames[split_idx:]

    train_ids = [iid for fn in train_files for iid in ids_by_filename[fn]]
    val_ids = [iid for fn in val_files for iid in ids_by_filename[fn]]

    return train_ids, val_ids


def get_dataloaders(
    image_dir: str,
    annotations_json: str,
    image_size: int = 512,
    batch_size: int = 4,
    num_workers: int = 2,
    train_ratio: float = 0.8,
    seed: int = 42,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Create high-performance train and validation DataLoaders."""
    train_ids, val_ids = create_data_splits(
        annotations_json, image_dir, train_ratio, seed
    )

    train_dataset = SolarFilamentDataset(
        image_dir=image_dir,
        annotations_json=annotations_json,
        image_size=image_size,
        augment=True,
        image_ids=train_ids,
        cache_dir="cache_512",
    )

    val_dataset = SolarFilamentDataset(
        image_dir=image_dir,
        annotations_json=annotations_json,
        image_size=image_size,
        augment=False,
        image_ids=val_ids,
        cache_dir="cache_512",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader
