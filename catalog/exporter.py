"""JSON and CSV catalog exporters."""
import csv
import json
from pathlib import Path
from typing import Dict, List


def export_catalog(records: List[Dict], output_dir: str | Path) -> tuple[Path, Path]:
    """Write filament_catalog.json and filament_catalog.csv to output_dir."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = directory / "filament_catalog.json", directory / "filament_catalog.csv"
    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    fields = ["image_id", "timestamp", "model", "model_name", "model_checkpoint", "threshold", "filament_id", "confidence", "area_px", "perimeter_px",
              "skeleton_length_px", "average_width_px", "sinuosity", "orientation_deg", "centroid",
              "bbox", "spatial_region", "physical",
              "aspect_ratio", "length_width_ratio", "perimeter_area_ratio", "skeleton_area_ratio", "compactness",
              "prob_min", "prob_max", "prob_std", "prob_median", "prob_p90",
              "disk_center_dist", "solar_coordinates", "solar_lat", "solar_lon",
              "active_region", "eruption_indicator", "filament_type", "rating", "orientation_stability"]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["centroid"] = json.dumps(row["centroid"])
            row["bbox"] = json.dumps(row["bbox"])
            row["physical"] = json.dumps(row["physical"])
            row["solar_coordinates"] = json.dumps(row.get("solar_coordinates"))
            writer.writerow(row)
    return json_path, csv_path
