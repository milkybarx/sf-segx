# Dataset Report: MAGFiLO 1.0

## Overview
- **Source**: Earth-Space AI Research Lab (www.esairlab.com)
- **License**: NSO/GONG
- **Version**: 1.0

## Statistics

| Property | Value |
|---|---|
| Training images (on disk) | 707 |
| Test images (on disk) | 180 |
| Total annotated images | 1154 |
| Total annotations | 8199 |
| Image dimensions | 2048 x 2048 |
| Image format | JPEG |
| Annotation format | COCO (polygons + bounding boxes + spines) |

## Category Distribution

| Category | Count | Percentage |
|---|---|---|
| Left | 2535 | 30.9% |
| Right | 2590 | 31.6% |
| Unidentifiable | 3074 | 37.5% |
| Ambiguous | 0 | 0.0% |

## Filament Area Statistics

| Metric | Value (pixels) |
|---|---|
| Minimum | 9 |
| Maximum | 37739 |
| Mean | 2119.8 |
| Median | 1228.0 |
| Std Dev | 2741.6 |

## Filaments per Image

| Metric | Value |
|---|---|
| Minimum | 1 |
| Maximum | 26 |
| Mean | 7.1 |

## Observations

1. **Images are H-alpha full-disk solar observations** from GONG/NSO network
2. **Solar filaments appear as DARK elongated structures** on the solar disk
3. **No "Ambiguous" category** annotations exist in the training set
4. **All annotations include polygon segmentation AND spine curves**
5. **Small filaments dominate**: median area is 1228 px (vs 2048x2048 = 4M px total)
6. **Class imbalance**: filaments cover very small fraction of the image
