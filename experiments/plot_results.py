"""
Plot Training Curves & Summary Report
======================================
Plots Loss, Dice, and IoU curves over epochs and prints a summary.
"""

import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_training_results(json_path: str = "experiments/training_results.json", output_dir: str = "outputs"):
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    history = data.get('history', [])
    if not history:
        print("No training history found in JSON.")
        return

    epochs = [h['epoch'] for h in history]
    train_loss = [h['train']['loss'] for h in history]
    val_loss = [h['val']['loss'] for h in history]

    train_dice = [h['train']['dice'] for h in history]
    val_dice = [h['val']['dice'] for h in history]

    val_iou = [h['val'].get('iou', 0) for h in history]
    val_precision = [h['val'].get('precision', 0) for h in history]
    val_recall = [h['val'].get('recall', 0) for h in history]

    # Find best epoch based on val_dice
    best_idx = max(range(len(val_dice)), key=lambda i: val_dice[i])
    best_epoch = epochs[best_idx]
    best_val_dice = val_dice[best_idx]
    best_val_iou = val_iou[best_idx]
    best_val_prec = val_precision[best_idx]
    best_val_rec = val_recall[best_idx]

    print("=" * 65)
    print(" SOLAR FILAMENT MODEL TRAINING SUMMARY")
    print("=" * 65)
    print(f" Total Epochs Completed: {len(epochs)}")
    print(f" Best Validation Epoch:  Epoch #{best_epoch}")
    print(f" Best Validation Dice:   {best_val_dice:.4f} ({best_val_dice*100:.2f}%)")
    print(f" Best Validation IoU:    {best_val_iou:.4f} ({best_val_iou*100:.2f}%)")
    print(f" Best Val Precision:     {best_val_prec:.4f} ({best_val_prec*100:.2f}%)")
    print(f" Best Val Recall:        {best_val_rec:.4f} ({best_val_rec*100:.2f}%)")
    print("=" * 65)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(epochs, train_loss, label='Train Loss', color='#3182CE', linewidth=2)
    axes[0].plot(epochs, val_loss, label='Val Loss', color='#E53E3E', linewidth=2, linestyle='--')
    axes[0].axvline(best_epoch, color='green', linestyle=':', label=f'Best Epoch ({best_epoch})')
    axes[0].set_title('Loss Curve (Focal + Soft-Dice)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend()

    # Dice & IoU plot
    axes[1].plot(epochs, train_dice, label='Train Dice', color='#3182CE', linewidth=2)
    axes[1].plot(epochs, val_dice, label='Val Dice', color='#38A169', linewidth=2)
    axes[1].plot(epochs, val_iou, label='Val IoU', color='#D69E2E', linewidth=2, linestyle='--')
    axes[1].axvline(best_epoch, color='green', linestyle=':', label=f'Best Epoch ({best_epoch})')
    axes[1].set_title('Segmentation Performance (Dice & IoU)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved training curves chart to: [{chart_path}]")


if __name__ == '__main__':
    plot_training_results()
