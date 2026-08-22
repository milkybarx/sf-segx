"""
Validation Threshold Sweep Optimization [0.24 to 0.50]
======================================================
Sweeps the decision threshold across the validation set to determine
the exact optimal Bayes operational threshold for maximum Dice score.

Usage:
    python experiments/threshold_sweep.py
"""

import os
import sys
import json
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.predict import SolarFilamentPredictor
from training.metrics import compute_all_metrics


def run_threshold_sweep(
    checkpoint_path: str = "checkpoints/phase2_hybrid_loss_dice0.7249.pth",
    thresholds: List[float] = None,
    output_dir: str = "outputs/threshold_sweep"
):
    os.makedirs(output_dir, exist_ok=True)
    if thresholds is None:
        thresholds = [0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print(f"[*] RUNNING VALIDATION THRESHOLD SWEEP ON: {checkpoint_path}")
    print(f"[*] Sweep Range: [{min(thresholds):.2f} to {max(thresholds):.2f}] | Step: 0.02")
    print("=" * 80)

    predictor = SolarFilamentPredictor(checkpoint_path=checkpoint_path)

    # Load validation sample images and masks from dataset
    from preprocessing.dataset import get_dataloaders
    _, val_loader = get_dataloaders(
        image_dir="images/MAGFiLO_1.0_Kaggle_2026/train/train_images",
        annotations_json="images/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json",
        image_size=512,
        batch_size=1,
        num_workers=0
    )

    print(f"[+] Loaded {len(val_loader)} validation ground-truth observations.")

    # 1. Collect all predicted continuous probability maps and ground-truth targets
    prob_maps = []
    gt_masks = []

    print("[*] Generating continuous calibrated probability maps across validation set...")
    with torch.no_grad():
        for idx, (img_tensor, mask_tensor) in enumerate(val_loader):
            if idx >= 60:  # Robust 60-image validation cohort
                break
            t_input = img_tensor.to(device)
            logits = predictor.model(t_input)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            target = mask_tensor.squeeze().cpu().numpy()

            prob_maps.append(probs)
            gt_masks.append(target)

    def eval_metrics_np(pred_binary, target_binary):
        intersection = np.logical_and(pred_binary, target_binary).sum()
        total_pred = pred_binary.sum()
        total_tgt = target_binary.sum()
        
        if total_pred == 0 and total_tgt == 0:
            return {'dice': 1.0, 'iou': 1.0, 'recall': 1.0, 'precision': 1.0}
        
        dice = (2.0 * intersection) / (total_pred + total_tgt + 1e-8)
        union = np.logical_or(pred_binary, target_binary).sum()
        iou = intersection / (union + 1e-8)
        recall = intersection / (total_tgt + 1e-8)
        precision = intersection / (total_pred + 1e-8)
        return {'dice': float(dice), 'iou': float(iou), 'recall': float(recall), 'precision': float(precision)}


    # 2. Evaluate metrics across all candidate thresholds
    sweep_results = []
    best_dice = -1.0
    best_thresh = 0.50

    print("-" * 80)
    print(f"{'Threshold (tau)':<16} | {'Val Dice':<12} | {'Val IoU':<12} | {'Recall':<12} | {'Precision':<12}")
    print("-" * 80)

    for tau in thresholds:
        dices, ious, recalls, precisions = [], [], [], []
        for p, tgt in zip(prob_maps, gt_masks):
            binary_pred = (p > tau).astype(np.uint8)
            binary_tgt = (tgt > 0.5).astype(np.uint8)
            m = eval_metrics_np(binary_pred, binary_tgt)
            dices.append(m['dice'])
            ious.append(m['iou'])
            recalls.append(m['recall'])
            precisions.append(m['precision'])

        mean_dice = float(np.mean(dices))
        mean_iou = float(np.mean(ious))
        mean_rec = float(np.mean(recalls))
        mean_prec = float(np.mean(precisions))

        if mean_dice > best_dice:
            best_dice = mean_dice
            best_thresh = tau
            best_mark = " <-- PEAK OPERATIONAL THRESHOLD"
        else:
            best_mark = ""

        print(f"{tau:<16.2f} | {mean_dice:<12.4f} | {mean_iou:<12.4f} | {mean_rec*100:<11.2f}% | {mean_prec*100:<11.2f}%{best_mark}")

        sweep_results.append({
            'threshold': tau,
            'val_dice': round(mean_dice, 4),
            'val_iou': round(mean_iou, 4),
            'recall': round(mean_rec, 4),
            'precision': round(mean_prec, 4)
        })

    print("=" * 80)
    print(f"[+] OPTIMAL DECISION THRESHOLD: tau = {best_thresh:.2f} (Peak Val Dice: {best_dice:.4f})")
    print("=" * 80)

    # Save JSON summary
    summary = {
        'checkpoint': checkpoint_path,
        'optimal_threshold': best_thresh,
        'peak_val_dice': best_dice,
        'sweep_data': sweep_results
    }
    json_path = os.path.join(output_dir, "threshold_sweep_results.json")
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # 3. Plot Threshold vs. Metrics Curves
    plt.figure(figsize=(9, 5.5), dpi=300)
    plt.style.use('dark_background')

    t_vals = [r['threshold'] for r in sweep_results]
    d_vals = [r['val_dice'] for r in sweep_results]
    i_vals = [r['val_iou'] for r in sweep_results]
    r_vals = [r['recall'] for r in sweep_results]
    p_vals = [r['precision'] for r in sweep_results]

    plt.plot(t_vals, d_vals, 'o-', color='#38BDF8', linewidth=2.5, label=f'Val Dice (Peak: {best_dice:.4f} @ tau={best_thresh:.2f})')
    plt.plot(t_vals, i_vals, 's--', color='#34D399', linewidth=2.0, label='Val IoU')
    plt.plot(t_vals, r_vals, '^-.', color='#F59E0B', linewidth=1.8, label='Recall (Faint Filaments)')
    plt.plot(t_vals, p_vals, 'v:', color='#EC4899', linewidth=1.8, label='Precision')

    plt.axvline(best_thresh, color='#EF4444', linestyle='--', linewidth=1.5, label=f'Optimal Threshold tau={best_thresh:.2f}')
    plt.title(f"Decision Threshold Sweep [0.24 - 0.50] vs. Segmentation Metrics\nOptimal tau = {best_thresh:.2f} (Peak Dice: {best_dice:.4f})", fontsize=12, fontweight='bold')
    plt.xlabel("Decision Threshold (tau)", fontsize=10)
    plt.ylabel("Metric Score [0.0 - 1.0]", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.legend(loc='lower left', fontsize=9)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "threshold_sweep_curve.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"[+] Saved Threshold Sweep Chart to: {plot_path}")
    print(f"[+] Saved Sweep Metrics JSON to: {json_path}")
    return summary


if __name__ == '__main__':
    run_threshold_sweep()
