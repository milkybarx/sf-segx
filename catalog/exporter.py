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
    fields = ["image_id", "timestamp", "model", "filament_id", "confidence", "area_px", "perimeter_px",
              "skeleton_length_px", "average_width_px", "sinuosity", "orientation_deg", "centroid",
              "bbox", "spatial_region", "physical"]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["centroid"] = json.dumps(row["centroid"])
            row["bbox"] = json.dumps(row["bbox"])
            row["physical"] = json.dumps(row["physical"])
            writer.writerow(row)
    return json_path, csv_path
