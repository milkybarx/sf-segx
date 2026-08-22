"""
Dataset Report Generator
=========================
Generates a comprehensive dataset analysis report with visualizations.
"""

import os
import sys
import json
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_dataset_report(project_root: str = None):
    """Generate comprehensive dataset analysis report."""
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    dataset_root = os.path.join(project_root, 'images', 'MAGFiLO_1.0_Kaggle_2026')
    annotations_path = os.path.join(dataset_root, 'train', 'MAGFiLO_1.0_Annotations_kaggle2026_train.json')
    train_dir = os.path.join(dataset_root, 'train', 'train_images')
    test_dir = os.path.join(dataset_root, 'test', 'test_images')
    output_dir = os.path.join(project_root, 'outputs', 'dataset_report')
    os.makedirs(output_dir, exist_ok=True)

    # Load annotations
    with open(annotations_path, 'r') as f:
        data = json.load(f)

    images = data['images']
    annotations = data['annotations']
    categories = data['categories']

    # Basic stats
    train_files = os.listdir(train_dir)
    test_files = os.listdir(test_dir)

    print("=" * 60)
    print("DATASET REPORT: MAGFiLO 1.0")
    print("=" * 60)
    print(f"Source: {data.get('info', {}).get('team', 'N/A')}")
    print(f"Version: {data.get('info', {}).get('version', 'N/A')}")
    print()
    print(f"Training images on disk: {len(train_files)}")
    print(f"Test images on disk:     {len(test_files)}")
    print(f"Total annotated images:  {len(images)}")
    print(f"Total annotations:       {len(annotations)}")
    print()

    # Image dimensions
    dims = Counter([(img['width'], img['height']) for img in images])
    print(f"Image dimensions: {dict(dims)}")

    # Categories
    print("\nCategories:")
    cat_dist = Counter([a['category_id'] for a in annotations])
    for cat in categories:
        count = cat_dist.get(cat['id'], 0)
        print(f"  {cat['id']}: {cat['name']} ({cat['supercategory']}) — {count} annotations")

    # Area statistics
    areas = [a['area'] for a in annotations]
    print(f"\nFilament area statistics:")
    print(f"  Min:    {min(areas):.0f} px")
    print(f"  Max:    {max(areas):.0f} px")
    print(f"  Mean:   {np.mean(areas):.1f} px")
    print(f"  Median: {np.median(areas):.1f} px")
    print(f"  Std:    {np.std(areas):.1f} px")

    # Filaments per image
    fils_per_img = Counter([a['image_id'] for a in annotations])
    counts = list(fils_per_img.values())
    print(f"\nFilaments per image:")
    print(f"  Min:    {min(counts)}")
    print(f"  Max:    {max(counts)}")
    print(f"  Mean:   {np.mean(counts):.1f}")

    # --- Visualizations ---

    # 1. Category distribution
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    cat_names = [c['name'] for c in categories]
    cat_counts = [cat_dist.get(c['id'], 0) for c in categories]
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#FFC107']
    axes[0].bar(cat_names, cat_counts, color=colors[:len(cat_names)])
    axes[0].set_title('Annotation Category Distribution')
    axes[0].set_ylabel('Count')
    for i, (name, count) in enumerate(zip(cat_names, cat_counts)):
        axes[0].text(i, count + 50, str(count), ha='center', fontweight='bold')

    # 2. Area distribution
    axes[1].hist(areas, bins=50, color='#2196F3', edgecolor='white', alpha=0.8)
    axes[1].set_title('Filament Area Distribution')
    axes[1].set_xlabel('Area (pixels)')
    axes[1].set_ylabel('Count')
    axes[1].axvline(np.mean(areas), color='red', linestyle='--', label=f'Mean={np.mean(areas):.0f}')
    axes[1].legend()

    # 3. Filaments per image
    axes[2].hist(counts, bins=range(0, max(counts) + 2), color='#4CAF50', edgecolor='white', alpha=0.8)
    axes[2].set_title('Filaments per Image')
    axes[2].set_xlabel('Number of filaments')
    axes[2].set_ylabel('Number of images')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dataset_statistics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved dataset_statistics.png")

    # 4. Sample images with annotations
    from preprocessing.dataset import load_coco_annotations, coco_poly_to_mask

    images_dict, annotations_by_image, _ = load_coco_annotations(annotations_path)

    # Find images that are on disk
    available = set(train_files)
    annotated_on_disk = [
        (iid, img) for iid, img in images_dict.items()
        if img['file_name'] in available and iid in annotations_by_image
    ]

    # Pick 6 samples with diverse filament counts
    annotated_on_disk.sort(key=lambda x: len(annotations_by_image[x[0]]), reverse=True)
    sample_indices = [0, len(annotated_on_disk)//4, len(annotated_on_disk)//2,
                      3*len(annotated_on_disk)//4, -2, -1]
    samples = [annotated_on_disk[i] for i in sample_indices]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (img_id, img_info) in enumerate(samples):
        img_path = os.path.join(train_dir, img_info['file_name'])
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Draw annotations
        anns = annotations_by_image.get(img_id, [])
        for ann in anns:
            seg = ann.get('segmentation', [])
            for poly in seg:
                pts = np.array(poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                cv2.polylines(img, [pts], True, (0, 0, 255), 2)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize for display
        display_size = 512
        img_display = cv2.resize(img_rgb, (display_size, display_size))

        axes[idx].imshow(img_display)
        axes[idx].set_title(f"{img_info['file_name']}\n{len(anns)} filaments", fontsize=9)
        axes[idx].axis('off')

    plt.suptitle('Sample Images with Ground Truth Annotations (Red Polygons)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_annotations.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved sample_annotations.png")

    # Generate markdown report
    report = f"""# Dataset Report: MAGFiLO 1.0

## Overview
- **Source**: {data.get('info', {}).get('team', 'N/A')}
- **License**: {data.get('licenses', {}).get('name', 'NSO/GONG')}
- **Version**: {data.get('info', {}).get('version', '1.0')}

## Statistics

| Property | Value |
|---|---|
| Training images (on disk) | {len(train_files)} |
| Test images (on disk) | {len(test_files)} |
| Total annotated images | {len(images)} |
| Total annotations | {len(annotations)} |
| Image dimensions | 2048 × 2048 |
| Image format | JPEG |
| Annotation format | COCO (polygons + bounding boxes + spines) |

## Category Distribution

| Category | Count | Percentage |
|---|---|---|
"""
    total_ann = len(annotations)
    for cat in categories:
        count = cat_dist.get(cat['id'], 0)
        pct = count / total_ann * 100
        report += f"| {cat['name']} | {count} | {pct:.1f}% |\n"

    report += f"""
## Filament Area Statistics

| Metric | Value (pixels) |
|---|---|
| Minimum | {min(areas):.0f} |
| Maximum | {max(areas):.0f} |
| Mean | {np.mean(areas):.1f} |
| Median | {np.median(areas):.1f} |
| Std Dev | {np.std(areas):.1f} |

## Filaments per Image

| Metric | Value |
|---|---|
| Minimum | {min(counts)} |
| Maximum | {max(counts)} |
| Mean | {np.mean(counts):.1f} |

## Observations

1. **Images are H-alpha full-disk solar observations** from GONG/NSO network
2. **Solar filaments appear as DARK elongated structures** on the solar disk
3. **No "Ambiguous" category** annotations exist in the training set
4. **All annotations include polygon segmentation AND spine curves**
5. **Small filaments dominate**: median area is {np.median(areas):.0f} px (vs 2048×2048 = 4M px total)
6. **Class imbalance**: filaments cover very small fraction of the image
"""

    report_path = os.path.join(project_root, 'dataset_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Saved dataset_report.md")

    return report


if __name__ == '__main__':
    generate_dataset_report()
