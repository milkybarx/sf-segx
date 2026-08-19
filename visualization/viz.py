"""
Visualization Utilities
=======================
Overlay generation, confidence maps, comparison plots, and training curve visualization.
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple


def create_filament_overlay(image: np.ndarray, mask: np.ndarray,
                             color: tuple = (255, 50, 50),
                             alpha: float = 0.4) -> np.ndarray:
    """Create semi-transparent colored overlay of filament mask on image."""
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    overlay = vis.copy()
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = color

    result = cv2.addWeighted(vis, 1 - alpha, overlay, alpha, 0)

    # Contour for clarity
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 255, 255), 1)

    return result


def probability_to_heatmap(prob_map: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Convert probability map to colored heatmap."""
    prob_uint8 = (prob_map * 255).clip(0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(prob_uint8, colormap)
    return heatmap


def create_confidence_visualization(prob_map: np.ndarray, image: np.ndarray = None,
                                      alpha: float = 0.5) -> np.ndarray:
    """Create confidence/probability visualization overlaid on image."""
    heatmap = probability_to_heatmap(prob_map)

    if image is not None:
        if len(image.shape) == 2:
            base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            base = image.copy()

        # Resize to match
        if base.shape[:2] != heatmap.shape[:2]:
            heatmap = cv2.resize(heatmap, (base.shape[1], base.shape[0]))

        result = cv2.addWeighted(base, 1 - alpha, heatmap, alpha, 0)
        return result

    return heatmap


def create_comparison_grid(results: Dict[str, np.ndarray],
                            target_size: int = 512) -> np.ndarray:
    """Create a grid of all intermediate results for comparison."""
    panels = []
    titles = [
        ('original', 'Original'),
        ('preprocessed', 'Preprocessed'),
        ('frangi_response', 'Frangi Response'),
        ('unet_probability', 'U-Net Prob'),
        ('final_mask', 'Final Mask'),
        ('overlay', 'Overlay'),
    ]

    for key, title in titles:
        img = results.get(key)
        if img is None:
            img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        else:
            # Resize
            img = cv2.resize(img, (target_size, target_size))

            # Convert to BGR for display
            if len(img.shape) == 2:
                if img.max() <= 1.0 and img.dtype in [np.float32, np.float64]:
                    img = (img * 255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Add title
        cv2.putText(img, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        panels.append(img)

    # Arrange in 2x3 grid
    row1 = np.hstack(panels[:3])
    row2 = np.hstack(panels[3:6]) if len(panels) >= 6 else np.hstack(panels[3:])
    if row1.shape[1] != row2.shape[1]:
        row2 = cv2.resize(row2, (row1.shape[1], row1.shape[0]))
    grid = np.vstack([row1, row2])

    return grid


def plot_training_curves(history: List[Dict], save_path: str = None) -> None:
    """Plot training and validation loss/metrics curves."""
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train']['loss'] for h in history]
    val_loss = [h['val']['loss'] for h in history]
    train_dice = [h['train']['dice'] for h in history]
    val_dice = [h['val']['dice'] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Dice
    axes[1].plot(epochs, train_dice, 'b-', label='Train Dice', linewidth=2)
    axes[1].plot(epochs, val_dice, 'r-', label='Val Dice', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Score')
    axes[1].set_title('Training & Validation Dice')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    plt.close()


def plot_method_comparison(results: Dict[str, Dict], save_path: str = None) -> None:
    """Plot bar chart comparing Frangi, U-Net, and Hybrid methods."""
    methods = list(results.keys())
    metrics_names = ['dice', 'iou', 'precision', 'recall']

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(metrics_names))
    width = 0.25

    for i, method in enumerate(methods):
        values = [results[method].get(m, 0) for m in metrics_names]
        bars = ax.bar(x + i * width, values, width, label=method)
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    ax.set_title('Segmentation Method Comparison')
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.capitalize() for m in metrics_names])
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    plt.close()
