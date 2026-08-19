"""Six-panel scientific visualization for Phase 2."""
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _rect_overlap_ratio(first, second) -> float:
    """Return intersection area relative to the smaller rectangle."""
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(1.0, (first[2] - first[0]) * (first[3] - first[1]))
    second_area = max(1.0, (second[2] - second[0]) * (second[3] - second[1]))
    return intersection / min(first_area, second_area)


def _place_label(label_rect, filament_rect, placed_labels, filament_rects,
                 image_width, image_height):
    """Find a compact nearby label position, falling back to a leader line."""
    label_width = label_rect[2] - label_rect[0]
    label_height = label_rect[3] - label_rect[1]
    gap = 6
    x1, y1, x2, y2 = filament_rect
    candidates = [
        (x1, y1 - label_height - gap),
        (x1, y2 + gap),
        (x2 + gap, y1),
        (x1 - label_width - gap, y1),
        (x1 - label_width - gap, y1 - label_height - gap),
        (x2 + gap, y1 - label_height - gap),
        (x1 - label_width - gap, y2 + gap),
        (x2 + gap, y2 + gap),
    ]

    def valid(candidate):
        candidate = (candidate[0], candidate[1], candidate[0] + label_width,
                     candidate[1] + label_height)
        if candidate[0] < 2 or candidate[1] < 2 or candidate[2] > image_width - 2 or candidate[3] > image_height - 2:
            return False
        if any(_rect_overlap_ratio(candidate, placed) > 0.0 for placed in placed_labels):
            return False
        # Do not put a label over another filament's scientific box.
        if any(_rect_overlap_ratio(candidate, other) > 0.05 for other in filament_rects):
            return False
        return True

    for candidate in candidates:
        if valid(candidate):
            return (candidate[0], candidate[1], candidate[0] + label_width,
                    candidate[1] + label_height), False

    # Search progressively farther around this filament before using a leader line.
    for distance in range(12, max(image_width, image_height), 12):
        expanded = [
            (x1, y1 - label_height - distance),
            (x1, y2 + distance),
            (x2 + distance, y1),
            (x1 - label_width - distance, y1),
        ]
        for candidate in expanded:
            if valid(candidate):
                return (candidate[0], candidate[1], candidate[0] + label_width,
                        candidate[1] + label_height), True
        if distance > max(image_width, image_height) // 2:
            break

    # The grid fallback guarantees a non-overlapping, unclipped label.
    for y in range(2, image_height - int(label_height) - 2, max(12, int(label_height))):
        for x in range(2, image_width - int(label_width) - 2, max(12, int(label_width))):
            if valid((x, y)):
                return (x, y, x + label_width, y + label_height), True
    return (2, 2, 2 + label_width, 2 + label_height), True


def _instance_panel(image: np.ndarray, filaments: List[Dict], skeleton: bool = False,
                    draw_boxes: bool = True, draw_labels: bool = True) -> np.ndarray:
    """Draw boxes, spines, and high-contrast labels in image coordinates."""
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if image.ndim == 2 else image.copy()
    image_height, image_width = rgb.shape[:2]
    placed_labels = []
    label_specs = []
    filament_rects = []
    for filament_index, filament in enumerate(filaments):
        bbox = filament["bbox"]
        source_width = filament.get("image_width", image_width)
        source_height = filament.get("image_height", image_height)
        scale_x = image_width / max(source_width, 1)
        scale_y = image_height / max(source_height, 1)
        filament_rects.append((
            bbox["x_min"] * scale_x, bbox["y_min"] * scale_y,
            bbox["x_max"] * scale_x, bbox["y_max"] * scale_y,
        ))
    for filament_index, filament in enumerate(filaments):
        if skeleton:
            spine = filament.get("skeleton_mask")
            if spine is not None:
                if spine.shape[:2] != (image_height, image_width):
                    spine = cv2.resize(spine.astype(np.uint8), (image_width, image_height),
                                       interpolation=cv2.INTER_NEAREST)
                rgb[spine > 0] = (0, 255, 255)
        bbox = filament["bbox"]
        source_width = filament.get("image_width", image_width)
        source_height = filament.get("image_height", image_height)
        scale_x = image_width / max(source_width, 1)
        scale_y = image_height / max(source_height, 1)
        x_min = int(round(bbox["x_min"] * scale_x))
        y_min = int(round(bbox["y_min"] * scale_y))
        x_max = int(round(bbox["x_max"] * scale_x))
        y_max = int(round(bbox["y_max"] * scale_y))
        x_min, x_max = np.clip([x_min, x_max], 0, image_width - 1)
        y_min, y_max = np.clip([y_min, y_max], 0, image_height - 1)
        if draw_boxes:
            cv2.rectangle(rgb, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 1)
        if not draw_labels:
            continue
        label = f"#{filament['filament_id']} | {float(filament.get('confidence', 0.0)):.3f} | {filament.get('spatial_region', 'CENTER')}"
        # At 200 DPI, 8 pt text is compact but remains sharp in the source-sized panel.
        label_width = max(112, len(label) * 7.0)
        label_height = 23
        label_rect, leader = _place_label(
            (0, 0, label_width, label_height),
            (x_min, y_min, x_max, y_max), placed_labels,
            filament_rects[:filament_index] + filament_rects[filament_index + 1:],
            image_width, image_height,
        )
        placed_labels.append(label_rect)
        label_specs.append((label_rect, label, (x_min, y_min, x_max, y_max), leader))

    figure = plt.figure(figsize=(image_width / 200, image_height / 200), dpi=200)
    axis = figure.add_axes([0, 0, 1, 1])
    axis.imshow(rgb, interpolation="nearest")
    axis.set_xlim(0, image_width)
    axis.set_ylim(image_height, 0)
    axis.axis("off")
    for placed, text, filament_rect, leader in label_specs:
        if leader:
            label_center = ((placed[0] + placed[2]) / 2, (placed[1] + placed[3]) / 2)
            filament_center = ((filament_rect[0] + filament_rect[2]) / 2,
                               (filament_rect[1] + filament_rect[3]) / 2)
            axis.plot([label_center[0], filament_center[0]], [label_center[1], filament_center[1]],
                      color="#f0f0f0", linewidth=0.7, zorder=4)
        axis.text(placed[0] + 5, placed[1] + 15, text, color="#39ff14", fontsize=8,
                  fontweight="bold", va="center", ha="left",
                  bbox={"boxstyle": "round,pad=0.18", "facecolor": "#050805",
                        "edgecolor": "#e5e5e5", "linewidth": 0.6, "alpha": 1.0})
    figure.canvas.draw()
    rendered = np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
    plt.close(figure)
    return rendered


def create_phase2_figure(image: np.ndarray, probability: np.ndarray, mask: np.ndarray,
                         filaments: List[Dict], attribution: Optional[np.ndarray] = None,
                         save_path: Optional[str | Path] = None):
    """Create original, probability, mask, instances, skeleton, and attribution panels."""
    panels = [image, probability, mask, _instance_panel(image, filaments), _instance_panel(image, filaments, True),
              attribution if attribution is not None else np.zeros_like(probability)]
    titles = ["Original", "Mask2Former probability", "Binary filament mask", "Instances + green boxes",
              "Skeletons + cyan spines", "Segmentation attribution"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=200)
    for axis, panel, title in zip(axes.flat, panels, titles):
        axis.imshow(panel, cmap="inferno" if panel.ndim == 2 and title != "Binary filament mask" else "gray")
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def save_filament_crops(image: np.ndarray, filaments: List[Dict], output_dir: str | Path,
                        padding: int = 12) -> List[Path]:
    """Save padded high-magnification crops for each detected filament."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for filament in filaments:
        bbox = filament["bbox"]
        x0, y0 = max(0, bbox["x_min"] - padding), max(0, bbox["y_min"] - padding)
        x1 = min(image.shape[1], bbox["x_max"] + padding)
        y1 = min(image.shape[0], bbox["y_max"] + padding)
        path = directory / f"filament_{filament['filament_id']:03d}.png"
        cv2.imwrite(str(path), image[y0:y1, x0:x1])
        paths.append(path)
    return paths
