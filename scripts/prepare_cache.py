"""
Precomputes and caches the GONGPreprocessor output (CLAHE-enhanced grayscale image +
solar-disk boolean mask) for every training image, so training doesn't redo this
CPU-bound work (contour finding, limb-darkening flattening, CLAHE) on every single
epoch. This is a pure caching layer -- output is bit-identical to calling
GONGPreprocessor.preprocess() directly, just computed once instead of once per epoch.
"""
import glob
import os

import cv2
import numpy as np
from tqdm import tqdm

from train import GONGPreprocessor, IMG_DIR

CACHE_DIR = os.path.join(os.path.dirname(IMG_DIR), "train_preprocessed")


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    preprocessor = GONGPreprocessor()

    image_paths = sorted(
        glob.glob(f"{IMG_DIR}/*.jpg") + glob.glob(f"{IMG_DIR}/*.jpeg") + glob.glob(f"{IMG_DIR}/*.png")
    )
    print(f"Caching preprocessed output for {len(image_paths)} images -> {CACHE_DIR}")

    for img_path in tqdm(image_paths):
        fn = os.path.splitext(os.path.basename(img_path))[0]
        enh_path = os.path.join(CACHE_DIR, fn + "_enh.png")
        disk_path = os.path.join(CACHE_DIR, fn + "_disk.png")
        if os.path.exists(enh_path) and os.path.exists(disk_path):
            continue

        raw_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        enhanced_img, disk_mask = preprocessor.preprocess(raw_img)
        cv2.imwrite(enh_path, enhanced_img)
        cv2.imwrite(disk_path, (disk_mask.astype(np.uint8) * 255))

    print("Done.")


if __name__ == "__main__":
    main()
