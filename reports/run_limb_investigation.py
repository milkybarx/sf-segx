"""
Solar Limb Filament Forensic Investigation Script
=================================================
Compares 0.93r vs 1.00r solar disk boundary on existing validation images with limb filaments.
Evaluates Model 3 and Model 5, ground-truth pixel overlap, and visual comparisons.
"""

import os
import sys
import json
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mask2former import build_mask2former
from preprocessing.dataset import load_coco_annotations, create_data_splits, coco_poly_to_mask
from preprocessing.solar_preprocessor import SolarPreprocessor


def run_limb_investigation():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Initializing Solar Limb Forensic Investigation on: {device}")

    img_dir = "images/MAGFiLO_1.0_Kaggle_2026/train/train_images"
    ann_file = "images/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"

    train_ids, val_ids = create_data_splits(ann_file, img_dir, train_ratio=0.8, seed=42)
    images_dict, annotations_by_image, _ = load_coco_annotations(ann_file)

    # 1. Load Model 3 (512px)
    ckpt3_path = "checkpoints/phase2_hybrid_loss_dice0.7249.pth"
    ckpt3 = torch.load(ckpt3_path, map_location=device, weights_only=False)
    m3_cfg = ckpt3.get('config', {}).get('model', {})
    model3 = build_mask2former(m3_cfg).to(device).eval()
    model3.load_state_dict(ckpt3['model_state_dict'])

    # 2. Load Model 5 (768px)
    ckpt5_path = "checkpoints/phase3_768res_dice0.7207.pth"
    ckpt5 = torch.load(ckpt5_path, map_location=device, weights_only=False)
    m5_cfg = ckpt5.get('config', {}).get('model', {})
    model5 = build_mask2former(m5_cfg).to(device).eval()
    model5.load_state_dict(ckpt5['model_state_dict'])

    # Preprocessors
    prep_curr_512 = SolarPreprocessor(target_size=512)
    prep_curr_768 = SolarPreprocessor(target_size=768)

    # Custom function for No-Erosion (1.00r)
    def preprocess_no_erosion(raw: np.ndarray, target_size: int = 512):
        gray = prep_curr_512.to_grayscale(raw)
        _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            h, w = gray.shape
            cx, cy, r = w // 2, h // 2, min(w, h) // 2
        else:
            largest = max(contours, key=cv2.contourArea)
            (cx, cy), radius = cv2.minEnclosingCircle(largest)
            cx, cy, r = int(cx), int(cy), int(radius * 1.00) # Full 1.00r disk

        disk_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(disk_mask, (cx, cy), r, 255, -1)

        corrected = prep_curr_512.correct_limb_darkening(gray, cx, cy, r)
        normalized = prep_curr_512.normalize(corrected, disk_mask)
        denoised = prep_curr_512.denoise(normalized, sigma=1.0)
        enhanced = prep_curr_512.enhance_contrast(denoised, clip_limit=2.0)
        enhanced[disk_mask == 0] = 0
        resized = cv2.resize(enhanced, (target_size, target_size))
        return resized.astype(np.float32) / 255.0, (cx, cy, r)

    # Output directory for visual comparisons
    out_dir = "outputs/limb_investigation"
    os.makedirs(out_dir, exist_ok=True)

    # Metrics storage
    total_gt_pixels_all = 0
    gt_pixels_beyond_093r = 0
    gt_pixels_limb_zone_080_093r = 0
    gt_pixels_interior = 0

    val_images_with_limb_filaments = []

    m3_curr_dices, m3_curr_ious = [], []
    m3_noer_dices, m3_noer_ious = [], []
    m5_curr_dices, m5_curr_ious = [], []
    m5_noer_dices, m5_noer_ious = [], []

    limb_m3_curr_dices, limb_m3_noer_dices = [], []
    limb_m5_curr_dices, limb_m5_noer_dices = [], []
    interior_m3_dices, interior_m5_dices = [], []

    print("[*] Processing all 231 validation images...")

    for idx, iid in enumerate(val_ids):
        fn = images_dict[iid]['file_name']
        raw = cv2.imread(os.path.join(img_dir, fn), cv2.IMREAD_GRAYSCALE)
        orig_h, orig_w = raw.shape[:2]

        # Ground truth raw polygon mask
        gt_raw = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for ann in annotations_by_image.get(iid, []):
            if ann.get('segmentation'):
                gt_raw = np.maximum(gt_raw, coco_poly_to_mask(ann['segmentation'], orig_h, orig_w))

        # Disk parameters on raw image
        _, binary = cv2.threshold(raw, 20, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            (cx_raw, cy_raw), radius_raw = cv2.minEnclosingCircle(largest)
        else:
            cx_raw, cy_raw, radius_raw = orig_w // 2, orig_h // 2, min(orig_h, orig_w) // 2

        # Radial distance map on raw image
        Y_raw, X_raw = np.ogrid[:orig_h, :orig_w]
        r_dist = np.sqrt((X_raw - cx_raw)**2 + (Y_raw - cy_raw)**2) / (radius_raw + 1e-5)

        gt_total = gt_raw.sum()
        total_gt_pixels_all += gt_total

        gt_beyond_093 = np.logical_and(gt_raw > 0, r_dist > 0.93).sum()
        gt_pixels_beyond_093r += gt_beyond_093

        gt_limb_080_093 = np.logical_and(gt_raw > 0, (r_dist >= 0.80) & (r_dist <= 0.93)).sum()
        gt_pixels_limb_zone_080_093r += gt_limb_080_093

        gt_interior = np.logical_and(gt_raw > 0, r_dist < 0.80).sum()
        gt_pixels_interior += gt_interior

        has_limb = (gt_limb_080_093 + gt_beyond_093) > 15
        if has_limb:
            val_images_with_limb_filaments.append({
                "id": iid,
                "file_name": fn,
                "gt_total": int(gt_total),
                "gt_limb": int(gt_limb_080_093 + gt_beyond_093),
                "gt_beyond_093r": int(gt_beyond_093)
            })

        # Preprocessing: Current (0.93r) vs No Erosion (1.00r)
        c0_curr_512 = prep_curr_512.preprocess_for_model(raw)
        c0_curr_768 = prep_curr_768.preprocess_for_model(raw)

        c0_noer_512, _ = preprocess_no_erosion(raw, target_size=512)
        c0_noer_768, _ = preprocess_no_erosion(raw, target_size=768)

        gt_512 = cv2.resize(gt_raw, (512, 512), interpolation=cv2.INTER_NEAREST)
        gt_768 = cv2.resize(gt_raw, (768, 768), interpolation=cv2.INTER_NEAREST)

        # 512px radial mask for evaluation
        Y_512, X_512 = np.ogrid[:512, :512]
        r_512 = np.sqrt((X_512 - 256)**2 + (Y_512 - 256)**2) / 256.0
        limb_mask_512 = (r_512 >= 0.80) & (r_512 <= 1.00)
        interior_mask_512 = r_512 < 0.80

        # Model 3 inference (Current 0.93r vs No Erosion 1.00r)
        t3_curr = torch.from_numpy(c0_curr_512).unsqueeze(0).unsqueeze(0).to(device)
        t3_noer = torch.from_numpy(c0_noer_512).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            p3_curr = (torch.sigmoid(model3(t3_curr)).squeeze().cpu().numpy() > 0.5).astype(np.uint8)
            p3_noer = (torch.sigmoid(model3(t3_noer)).squeeze().cpu().numpy() > 0.5).astype(np.uint8)

        # Model 5 inference (Current 0.93r vs No Erosion 1.00r)
        t5_curr = torch.from_numpy(c0_curr_768).unsqueeze(0).unsqueeze(0).to(device)
        t5_noer = torch.from_numpy(c0_noer_768).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            p5_curr_raw = (torch.sigmoid(model5(t5_curr)).squeeze().cpu().numpy() > 0.5).astype(np.uint8)
            p5_noer_raw = (torch.sigmoid(model5(t5_noer)).squeeze().cpu().numpy() > 0.5).astype(np.uint8)
        p5_curr = cv2.resize(p5_curr_raw, (512, 512), interpolation=cv2.INTER_NEAREST)
        p5_noer = cv2.resize(p5_noer_raw, (512, 512), interpolation=cv2.INTER_NEAREST)

        def dice_iou(p, g):
            inter = np.logical_and(p, g).sum()
            tot = p.sum() + g.sum()
            un = np.logical_or(p, g).sum()
            d = (2.0 * inter) / (tot + 1e-7) if tot > 0 else (1.0 if p.sum() == 0 and g.sum() == 0 else 0.0)
            i = inter / (un + 1e-7) if un > 0 else (1.0 if p.sum() == 0 and g.sum() == 0 else 0.0)
            return float(d), float(i)

        # Whole disk metrics
        d3_c, i3_c = dice_iou(p3_curr, gt_512)
        d3_n, i3_n = dice_iou(p3_noer, gt_512)
        d5_c, i5_c = dice_iou(p5_curr, gt_512)
        d5_n, i5_n = dice_iou(p5_noer, gt_512)

        m3_curr_dices.append(d3_c); m3_curr_ious.append(i3_c)
        m3_noer_dices.append(d3_n); m3_noer_ious.append(i3_n)
        m5_curr_dices.append(d5_c); m5_curr_ious.append(i5_c)
        m5_noer_dices.append(d5_n); m5_noer_ious.append(i5_n)

        # Limb zone metrics
        if np.logical_and(gt_512, limb_mask_512).sum() > 5:
            gt_l = np.logical_and(gt_512, limb_mask_512)
            d3_l_c, _ = dice_iou(np.logical_and(p3_curr, limb_mask_512), gt_l)
            d3_l_n, _ = dice_iou(np.logical_and(p3_noer, limb_mask_512), gt_l)
            d5_l_c, _ = dice_iou(np.logical_and(p5_curr, limb_mask_512), gt_l)
            d5_l_n, _ = dice_iou(np.logical_and(p5_noer, limb_mask_512), gt_l)
            limb_m3_curr_dices.append(d3_l_c)
            limb_m3_noer_dices.append(d3_l_n)
            limb_m5_curr_dices.append(d5_l_c)
            limb_m5_noer_dices.append(d5_l_n)

        # Interior zone metrics
        if np.logical_and(gt_512, interior_mask_512).sum() > 5:
            gt_int = np.logical_and(gt_512, interior_mask_512)
            d3_int, _ = dice_iou(np.logical_and(p3_curr, interior_mask_512), gt_int)
            d5_int, _ = dice_iou(np.logical_and(p5_curr, interior_mask_512), gt_int)
            interior_m3_dices.append(d3_int)
            interior_m5_dices.append(d5_int)

    # 4. Generate 6 representative visual comparisons for limb validation images
    print(f"[*] Found {len(val_images_with_limb_filaments)} validation images with limb filaments. Generating visual comparisons...")
    
    for v_idx, limb_info in enumerate(val_images_with_limb_filaments[:6]):
        fn = limb_info['file_name']
        raw = cv2.imread(os.path.join(img_dir, fn), cv2.IMREAD_GRAYSCALE)
        orig_h, orig_w = raw.shape[:2]

        gt_raw = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for ann in annotations_by_image.get(limb_info['id'], []):
            if ann.get('segmentation'):
                gt_raw = np.maximum(gt_raw, coco_poly_to_mask(ann['segmentation'], orig_h, orig_w))

        # Disk overlays
        raw_bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        _, binary = cv2.threshold(raw, 20, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            (cx_raw, cy_raw), radius_raw = cv2.minEnclosingCircle(largest)
            cx_raw, cy_raw, r_raw = int(cx_raw), int(cy_raw), int(radius_raw)
            cv2.circle(raw_bgr, (cx_raw, cy_raw), r_raw, (0, 255, 0), 2)         # Green = 1.00r
            cv2.circle(raw_bgr, (cx_raw, cy_raw), int(r_raw*0.93), (0, 0, 255), 2) # Red = 0.93r

        c0_curr = prep_curr_512.preprocess_for_model(raw)
        c0_noer, _ = preprocess_no_erosion(raw, target_size=512)
        c0_curr_768 = prep_curr_768.preprocess_for_model(raw)
        c0_noer_768, _ = preprocess_no_erosion(raw, target_size=768)

        gt_512 = cv2.resize(gt_raw, (512, 512), interpolation=cv2.INTER_NEAREST)

        t3_c = torch.from_numpy(c0_curr).unsqueeze(0).unsqueeze(0).to(device)
        t3_n = torch.from_numpy(c0_noer).unsqueeze(0).unsqueeze(0).to(device)
        t5_c = torch.from_numpy(c0_curr_768).unsqueeze(0).unsqueeze(0).to(device)
        t5_n = torch.from_numpy(c0_noer_768).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            p3_c = (torch.sigmoid(model3(t3_c)).squeeze().cpu().numpy() > 0.5).astype(np.uint8)
            p3_n = (torch.sigmoid(model3(t3_n)).squeeze().cpu().numpy() > 0.5).astype(np.uint8)
            p5_c = cv2.resize((torch.sigmoid(model5(t5_c)).squeeze().cpu().numpy() > 0.5).astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST)
            p5_n = cv2.resize((torch.sigmoid(model5(t5_n)).squeeze().cpu().numpy() > 0.5).astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST)

        # 6-Panel Figure
        fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=140)
        fig.patch.set_facecolor('#0f172a')

        # Row 1: Raw with boundaries, GT, Current Preprocessing (0.93r)
        axes[0, 0].imshow(cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title(f"Raw Image + Disks\nGreen: 1.00r | Red: 0.93r (Removed)", color='#38bdf8', fontsize=11)
        axes[0, 0].axis('off')

        axes[0, 1].imshow(gt_512, cmap='Blues')
        axes[0, 1].set_title(f"Ground Truth Mask\n(Limb GT Pixels: {limb_info['gt_limb']})", color='#4ade80', fontsize=11)
        axes[0, 1].axis('off')

        axes[0, 2].imshow(c0_curr, cmap='gray')
        axes[0, 2].set_title("Current Preprocessing (0.93r Masked)", color='#e2e8f0', fontsize=11)
        axes[0, 2].axis('off')

        # Row 2: No-Erosion Preprocessing (1.00r), Model 3 Prediction (0.93r vs 1.00r), Model 5 Prediction
        axes[1, 0].imshow(c0_noer, cmap='gray')
        axes[1, 0].set_title("No-Erosion Preproc (Full 1.00r Disk)", color='#e2e8f0', fontsize=11)
        axes[1, 0].axis('off')

        d3_c_val, _ = dice_iou(p3_c, gt_512)
        d3_n_val, _ = dice_iou(p3_n, gt_512)
        axes[1, 1].imshow(p3_c, cmap='Blues')
        axes[1, 1].set_title(f"Model 3 Prediction\n0.93r Dice: {d3_c_val:.3f} | 1.00r: {d3_n_val:.3f}", color='#a78bfa', fontsize=11)
        axes[1, 1].axis('off')

        d5_c_val, _ = dice_iou(p5_c, gt_512)
        d5_n_val, _ = dice_iou(p5_n, gt_512)
        axes[1, 2].imshow(p5_c, cmap='Blues')
        axes[1, 2].set_title(f"Model 5 (768px) Prediction\n0.93r Dice: {d5_c_val:.3f} | 1.00r: {d5_n_val:.3f}", color='#fbbf24', fontsize=11)
        axes[1, 2].axis('off')

        plt.tight_layout()
        save_path = os.path.join(out_dir, f"limb_compare_{v_idx+1:02d}_{os.path.splitext(fn)[0]}.png")
        plt.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close()

    # Raw results summary
    results_summary = {
        "dataset_analysis": {
            "total_gt_filament_pixels_in_val": int(total_gt_pixels_all),
            "gt_pixels_beyond_093r": int(gt_pixels_beyond_093r),
            "gt_pixels_beyond_093r_percentage": float(gt_pixels_beyond_093r / (total_gt_pixels_all + 1e-7) * 100),
            "gt_pixels_in_limb_zone_080_093r": int(gt_pixels_limb_zone_080_093r),
            "gt_pixels_in_limb_zone_080_093r_percentage": float(gt_pixels_limb_zone_080_093r / (total_gt_pixels_all + 1e-7) * 100),
            "gt_pixels_in_interior_under_080r": int(gt_pixels_interior),
            "gt_pixels_in_interior_under_080r_percentage": float(gt_pixels_interior / (total_gt_pixels_all + 1e-7) * 100),
            "val_images_with_limb_filaments_count": len(val_images_with_limb_filaments)
        },
        "performance_comparison": {
            "interior_filaments": {
                "model3_dice": float(np.mean(interior_m3_dices)),
                "model5_dice": float(np.mean(interior_m5_dices))
            },
            "limb_filaments_zone_080_100r": {
                "model3_curr_093r_dice": float(np.mean(limb_m3_curr_dices)),
                "model3_no_erosion_100r_dice": float(np.mean(limb_m3_noer_dices)),
                "model5_curr_093r_dice": float(np.mean(limb_m5_curr_dices)),
                "model5_no_erosion_100r_dice": float(np.mean(limb_m5_noer_dices))
            },
            "whole_disk": {
                "model3_curr_093r_dice": float(np.mean(m3_curr_dices)),
                "model3_no_erosion_100r_dice": float(np.mean(m3_noer_dices)),
                "model5_curr_093r_dice": float(np.mean(m5_curr_dices)),
                "model5_no_erosion_100r_dice": float(np.mean(m5_noer_dices))
            }
        }
    }

    with open(os.path.join(out_dir, "limb_forensic_results.json"), 'w') as f:
        json.dump(results_summary, f, indent=2)

    print("[+] Limb Forensic Investigation complete. Results saved to outputs/limb_investigation/")
    print(json.dumps(results_summary, indent=2))

if __name__ == '__main__':
    run_limb_investigation()
