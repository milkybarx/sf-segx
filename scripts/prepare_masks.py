"""
Builds binary filament segmentation masks (PNG, same resolution as source
images) from the MAGFiLO COCO-style polygon annotations shipped with the
Kaggle "filament-segmentation-2026" competition.

Each physical image may appear multiple times in the JSON's "images" list
(one entry per annotation session). Sessions for the same file disagree
substantially (measured: union area ~2.4x intersection area on a sample --
these are not near-duplicate consensus labels), so unioning them produces a
noisy, inflated target. Instead we use a single canonical session per file
(the first one) for a self-consistent ground-truth mask.
"""
import json
import os

import cv2
import numpy as np
from tqdm import tqdm

DATA_ROOT = "data/MAGFiLO_1.0_Kaggle_2026"
ANN_PATH = os.path.join(DATA_ROOT, "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
IMG_DIR = os.path.join(DATA_ROOT, "train/train_images")
MASK_DIR = os.path.join(DATA_ROOT, "train/train_masks")


def polygons_to_mask(segmentations, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    for seg in segmentations:
        if not isinstance(seg, list) or len(seg) < 6:
            continue
        pts = np.array(seg, dtype=np.float64).reshape(-1, 2)
        pts = np.round(pts).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def main():
    os.makedirs(MASK_DIR, exist_ok=True)

    with open(ANN_PATH, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # image_id -> file_name / height / width
    images_by_id = {im["id"]: im for im in coco["images"]}

    # file_name -> list of image_ids that reference it (multiple annotator sessions)
    ids_by_filename = {}
    for im in coco["images"]:
        ids_by_filename.setdefault(im["file_name"], []).append(im["id"])

    # image_id -> list of segmentation polygon-lists
    segs_by_image_id = {}
    skipped_rle = 0
    for ann in coco["annotations"]:
        seg = ann.get("segmentation")
        if isinstance(seg, dict):  # RLE-encoded crowd annotation, none expected here
            skipped_rle += 1
            continue
        segs_by_image_id.setdefault(ann["image_id"], []).extend(seg)

    print(f"Unique files: {len(ids_by_filename)} | image records: {len(coco['images'])} "
          f"| annotations: {len(coco['annotations'])} | skipped RLE anns: {skipped_rle}")

    n_written, n_empty = 0, 0
    for file_name, img_ids in tqdm(ids_by_filename.items(), desc="Rasterizing masks"):
        canonical_id = sorted(img_ids)[0]
        ref = images_by_id[canonical_id]
        h, w = ref["height"], ref["width"]

        polys = segs_by_image_id.get(canonical_id, [])
        mask = polygons_to_mask(polys, h, w)
        if mask.sum() == 0:
            n_empty += 1

        out_path = os.path.join(MASK_DIR, os.path.splitext(file_name)[0] + ".png")
        cv2.imwrite(out_path, mask * 255)
        n_written += 1

    print(f"Wrote {n_written} masks to {MASK_DIR} ({n_empty} empty)")


if __name__ == "__main__":
    main()
