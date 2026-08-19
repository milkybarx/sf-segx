"""
Pre-build Dataset Cache
========================
Pre-processes and resizes all solar images and polygon masks to 512x512.
Enables instant training epochs on the GPU.
"""

import os
import sys
import json
import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.solar_preprocessor import SolarPreprocessor
from preprocessing.dataset import load_coco_annotations, coco_poly_to_mask


def build_cache(
    image_dir: str = "images/MAGFiLO_1.0_Kaggle_2026/train/train_images",
    annotations_json: str = "images/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json",
    cache_dir: str = "cache_512",
    target_size: int = 512,
):
    os.makedirs(cache_dir, exist_ok=True)
    preprocessor = SolarPreprocessor(target_size=target_size)

    images_dict, annotations_by_image, _ = load_coco_annotations(annotations_json)
    available_files = set(os.listdir(image_dir))

    valid_items = [
        (iid, img) for iid, img in images_dict.items()
        if img['file_name'] in available_files and iid in annotations_by_image
    ]

    print(f"Building 512x512 cache for {len(valid_items)} images into [{cache_dir}]...")

    for img_id, img_info in tqdm(valid_items, desc="Pre-caching dataset"):
        file_name = img_info['file_name']
        base_id = os.path.splitext(file_name)[0]

        img_save_path = os.path.join(cache_dir, f"{base_id}_img.npy")
        mask_save_path = os.path.join(cache_dir, f"{base_id}_mask.npy")

        if os.path.exists(img_save_path) and os.path.exists(mask_save_path):
            continue

        raw_path = os.path.join(image_dir, file_name)
        raw = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            continue

        orig_h, orig_w = raw.shape[:2]

        # Polygon mask
        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for ann in annotations_by_image.get(img_id, []):
            seg = ann.get('segmentation', [])
            if seg:
                ann_mask = coco_poly_to_mask(seg, orig_h, orig_w)
                mask = np.maximum(mask, ann_mask)

        # Preprocess
        preproc = preprocessor.preprocess_for_model(raw)
        mask_resized = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST).astype(np.float32)

        np.save(img_save_path, preproc)
        np.save(mask_save_path, mask_resized)

    print(f"Cache complete! Cached files located in [{cache_dir}].")


if __name__ == "__main__":
    build_cache()
