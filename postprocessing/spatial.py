"""Image-plane bounding boxes and spatial region tags."""
from typing import Dict


def spatial_region(centroid: Dict[str, float], width: int, height: int) -> str:
    """Assign one of nine image-plane regions; these are not heliographic coordinates."""
    x, y = centroid["x"], centroid["y"]
    col = 0 if x < width / 3 else 2 if x >= 2 * width / 3 else 1
    row = 0 if y < height / 3 else 2 if y >= 2 * height / 3 else 1
    return {(0, 0): "NW", (1, 0): "N", (2, 0): "NE", (0, 1): "W",
            (1, 1): "CENTER", (2, 1): "E", (0, 2): "SW", (1, 2): "S",
            (2, 2): "SE"}[(col, row)]


def add_spatial_metadata(filament: Dict, image_shape: tuple) -> Dict:
    """Normalize the existing bbox and append image-plane spatial tagging."""
    height, width = image_shape[:2]
    bbox = filament["bbox"]
    x, y = int(bbox.get("x", bbox.get("x_min", 0))), int(bbox.get("y", bbox.get("y_min", 0)))
    w, h = int(bbox.get("width", bbox.get("x_max", x) - x)), int(bbox.get("height", bbox.get("y_max", y) - y))
    filament["bbox"] = {"x_min": x, "y_min": y, "x_max": x + w, "y_max": y + h,
                         "width": w, "height": h}
    filament["spatial_region"] = spatial_region(filament["centroid"], width, height)
    return filament
