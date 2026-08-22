"""
Evaluation & Method Comparison
==============================
Compare Frangi-only, U-Net-only, and Hybrid approaches on the validation set.
Generates comparison report and visualizations.
"""

import os
import sys
import json
import time
import yaml
import numpy as np
import cv2
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet import build_unet
from preprocessing.dataset import SolarFilamentDataset, create_data_splits, load_coco_annotations
from preprocessing.solar_preprocessor import SolarPreprocessor
from classical.frangi import FrangiPipeline
from hybrid.fusion import fuse_predictions, sweep_alpha
from training.metrics import compute_metrics_numpy
from visualization.viz import plot_method_comparison, plot_training_curves


def evaluate_all_methods(config_path: str = None):
    """Run full evaluation comparing all three methods."""

    # Load config
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    'configs', 'default_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_root = os.path.join(project_root, config['data']['dataset_root'])
    image_dir = os.path.join(dataset_root, config['data']['train_images_dir'])
    annotations_json = os.path.join(dataset_root, config['data']['annotations_file'])
    checkpoint_dir = os.path.join(project_root, config['output']['checkpoint_dir'])
    experiment_dir = os.path.join(project_root, config['output']['experiment_dir'])
    vis_dir = os.path.join(project_root, config['output']['visualization_dir'])
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    image_size = config['data']['image_size']

    # Setup device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Data split — get validation IDs
    _, val_ids = create_data_splits(
        annotations_json, image_dir,
        train_ratio=config['data']['train_ratio'],
        seed=config['data']['seed']
    )

    # Load validation dataset
    val_dataset = SolarFilamentDataset(
        image_dir=image_dir,
        annotations_json=annotations_json,
        image_size=image_size,
        augment=False,
        image_ids=val_ids,
    )

    # Load U-Net model
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
    if os.path.exists(checkpoint_path):
        model = build_unet(config.get('model', {}))
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        print(f"Loaded model from {checkpoint_path}")
        has_unet = True
    else:
        print(f"WARNING: No checkpoint at {checkpoint_path}. U-Net evaluation skipped.")
        has_unet = False

    # Frangi pipeline
    frangi_cfg = config.get('frangi', {})
    frangi = FrangiPipeline(
        scales=frangi_cfg.get('scales', [1, 2, 3, 5, 8]),
        alpha=frangi_cfg.get('alpha', 0.5),
        beta=frangi_cfg.get('beta', 0.5),
        gamma=frangi_cfg.get('gamma', 15.0),
        threshold=frangi_cfg.get('threshold', 0.15),
        min_area=frangi_cfg.get('min_area', 100),
        max_area=frangi_cfg.get('max_area', 50000),
        target_size=image_size,
    )

    preprocessor = SolarPreprocessor(target_size=image_size)

    # Collect predictions
    print(f"\nEvaluating on {len(val_dataset)} validation images...")

    frangi_metrics_list = []
    unet_metrics_list = []
    unet_probs_list = []
    frangi_probs_list = []
    gt_masks_list = []

    frangi_times = []
    unet_times = []

    for idx in tqdm(range(len(val_dataset)), desc="Evaluating"):
        image_tensor, mask_tensor = val_dataset[idx]

        # Ground truth mask
        gt_mask = mask_tensor.squeeze().numpy()
        gt_masks_list.append(gt_mask)

        # Get original image for Frangi
        image_id = val_dataset.image_ids[idx]
        img_info = val_dataset.images_dict[image_id]
        img_path = os.path.join(image_dir, img_info['file_name'])
        original_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # --- Frangi ---
        t0 = time.time()
        frangi_results = frangi.process_resized(original_image)
        frangi_time = time.time() - t0
        frangi_times.append(frangi_time)

        frangi_mask = frangi_results.get('filament_mask', np.zeros_like(gt_mask))
        frangi_prob = frangi_results.get('frangi_probability', np.zeros_like(gt_mask))

        # Resize to match ground truth
        if frangi_mask.shape != gt_mask.shape:
            frangi_mask = cv2.resize(frangi_mask, (gt_mask.shape[1], gt_mask.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
            frangi_prob = cv2.resize(frangi_prob, (gt_mask.shape[1], gt_mask.shape[0]),
                                      interpolation=cv2.INTER_LINEAR)

        frangi_m = compute_metrics_numpy(frangi_mask, gt_mask)
        frangi_metrics_list.append(frangi_m)
        frangi_probs_list.append(frangi_prob)

        # --- U-Net ---
        if has_unet:
            t0 = time.time()
            with torch.no_grad():
                input_tensor = image_tensor.unsqueeze(0).to(device)
                logits = model(input_tensor)
                unet_prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            unet_time = time.time() - t0
            unet_times.append(unet_time)

            unet_mask = (unet_prob > 0.5).astype(np.uint8)
            unet_m = compute_metrics_numpy(unet_mask, gt_mask)
            unet_metrics_list.append(unet_m)
            unet_probs_list.append(unet_prob)

    # Aggregate metrics
    def avg_metrics(metrics_list):
        if not metrics_list:
            return {}
        avg = {}
        for key in metrics_list[0]:
            avg[key] = np.mean([m[key] for m in metrics_list])
        return avg

    results = {}

    # Frangi results
    frangi_avg = avg_metrics(frangi_metrics_list)
    frangi_avg['avg_time_ms'] = np.mean(frangi_times) * 1000
    results['Frangi'] = frangi_avg
    print(f"\nFrangi:  Dice={frangi_avg['dice']:.4f} IoU={frangi_avg['iou']:.4f} "
          f"P={frangi_avg['precision']:.4f} R={frangi_avg['recall']:.4f} "
          f"Time={frangi_avg['avg_time_ms']:.1f}ms")

    # U-Net results
    if has_unet:
        unet_avg = avg_metrics(unet_metrics_list)
        unet_avg['avg_time_ms'] = np.mean(unet_times) * 1000
        results['U-Net'] = unet_avg
        print(f"U-Net:   Dice={unet_avg['dice']:.4f} IoU={unet_avg['iou']:.4f} "
              f"P={unet_avg['precision']:.4f} R={unet_avg['recall']:.4f} "
              f"Time={unet_avg['avg_time_ms']:.1f}ms")

        # Hybrid sweep
        print("\nHybrid fusion alpha sweep:")
        alphas = config.get('fusion', {}).get('alphas', [0.0, 0.25, 0.5, 0.75, 1.0])
        hybrid_results = sweep_alpha(unet_probs_list, frangi_probs_list, gt_masks_list, alphas)

        # Find best alpha
        best_alpha = max(hybrid_results, key=lambda a: hybrid_results[a]['dice'])
        best_hybrid = hybrid_results[best_alpha]
        results[f'Hybrid (α={best_alpha:.2f})'] = best_hybrid
        print(f"\nBest hybrid α={best_alpha:.2f}: Dice={best_hybrid['dice']:.4f}")

    # Save results
    # Convert float keys to strings for JSON
    save_results = {}
    for method, metrics in results.items():
        save_results[method] = {k: float(v) for k, v in metrics.items()}

    results_path = os.path.join(experiment_dir, 'comparison_results.json')
    with open(results_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Plot comparison
    plot_method_comparison(results, os.path.join(vis_dir, 'method_comparison.png'))

    # Plot training curves if available
    training_results_path = os.path.join(experiment_dir, 'training_results.json')
    if os.path.exists(training_results_path):
        with open(training_results_path, 'r') as f:
            training_data = json.load(f)
        plot_training_curves(training_data['history'],
                              os.path.join(vis_dir, 'training_curves.png'))

    return results


if __name__ == '__main__':
    evaluate_all_methods()
