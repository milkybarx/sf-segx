"""Generate manual tracking audit visualizations for Phase 2D."""
import os
import sys
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import json

# Add root folder to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.coordinates import pixel_to_stonyhurst

def main():
    print("Generating Phase 2D Tracking Audit Visualizations...")
    
    forecast_path = Path("data/training/filament_forecast_full.csv")
    if not forecast_path.exists():
        print("Forecast table not found. Run Phase 2D extraction first.")
        return
        
    df_forecast = pd.read_csv(forecast_path)
    
    # Select a few distinct tracks that have length >= 3
    track_counts = df_forecast["physical_track_id"].value_counts()
    long_tracks = track_counts[track_counts >= 3].index.tolist()
    
    if not long_tracks:
        print("No tracks with length >= 3 found.")
        return
        
    # Select top 5 longest tracks
    selected_tracks = long_tracks[:5]
    
    out_dir = Path("visualizations/phase2d_tracking")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    image_dir = Path("data/MAGFiLO_1.0_Kaggle_2026/train/train_images")
    mask_dir = Path("data/MAGFiLO_1.0_Kaggle_2026/train/train_masks")
    
    for t_id in selected_tracks:
        print(f"Visualizing track {t_id}...")
        track_df = df_forecast[df_forecast["physical_track_id"] == t_id].sort_values("observation_time")
        
        for idx, row in track_df.iterrows():
            img_id = row["image_id"]
            img_path = image_dir / f"{img_id}.jpeg"
            mask_path = mask_dir / f"{img_id}.png"
            
            if not img_path.exists():
                img_path = image_dir / f"{img_id}.jpg"
                
            if not img_path.exists() or not mask_path.exists():
                continue
                
            raw = cv2.imread(str(img_path))
            if raw is None:
                continue
                
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask = (mask > 127).astype(np.uint8)
                raw[mask > 0] = [0, 0, 255] # Red overlay
                
            # Add text
            time_str = row["observation_time"]
            lat = row["centroid_lat"]
            lon = row["centroid_lon"]
            
            text = f"Track {t_id} | Time {time_str} | Lat {lat:.1f} Lon {lon:.1f}"
            cv2.putText(raw, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            out_file = out_dir / f"track_{t_id}_{img_id}.jpg"
            cv2.imwrite(str(out_file), raw)
            
    print(f"Tracking visualizations saved to {out_dir}")

if __name__ == "__main__":
    main()
