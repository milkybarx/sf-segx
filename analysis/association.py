"""Filament-to-Flare Association Engine using NASA DONKI events."""
import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from pathlib import Path
from analysis.coordinates import parse_stonyhurst_string

logger = logging.getLogger("association")

DEFAULT_WEIGHTS = {
    "active_region_match": 0.40,
    "temporal_score": 0.25,
    "spatial_score": 0.20,
    "eruption_indicator": 0.10,
    "directional_consistency": 0.05
}

class FlareAssociationEngine:
    """Computes probabilistic associations between detected filaments and DONKI flares."""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None,
                 high_thresh: float = 0.70, med_thresh: float = 0.40, low_thresh: float = 0.15):
        self.weights = weights or DEFAULT_WEIGHTS
        self.high_thresh = high_thresh
        self.med_thresh = med_thresh
        self.low_thresh = low_thresh
        
        self.links_dir = Path("data/links")
        self.links_dir.mkdir(parents=True, exist_ok=True)
        self.links_csv = self.links_dir / "filament_flare_links.csv"

    def compute_temporal_score(self, fil_time_str: str, flr_time_str: str) -> float:
        """Score based on time difference. Positive difference means flare starts after filament observation."""
        try:
            # Clean timestamps
            t_fil = datetime.fromisoformat(fil_time_str.replace('Z', ''))
            t_flr = datetime.fromisoformat(flr_time_str.replace('Z', ''))
            
            dt = (t_flr - t_fil).total_seconds() / 3600.0  # in hours
            
            # Normal flare-triggering window: flare starts 0 to 12 hours after filament observation
            if 0.0 <= dt <= 6.0:
                return 1.0
            elif -2.0 <= dt < 0.0 or 6.0 < dt <= 12.0:
                return 0.7
            elif 12.0 < dt <= 24.0:
                return 0.3
            elif -6.0 <= dt < -2.0:
                return 0.2
            else:
                return 0.0
        except Exception as e:
            logger.error(f"Error parsing timestamps: {e}")
            return 0.0

    def compute_spatial_score(self, fil_lat: float, fil_lon: float, flr_loc_str: Optional[str]) -> float:
        """Score based on great-circle angular distance on the solar sphere in degrees."""
        if np.isnan(fil_lat) or np.isnan(fil_lon) or not flr_loc_str:
            return float("nan")
            
        flr_lat, flr_lon = parse_stonyhurst_string(flr_loc_str)
        if flr_lat is None or flr_lon is None:
            return float("nan")
            
        # Spherical distance
        lat1, lon1 = np.radians(fil_lat), np.radians(fil_lon)
        lat2, lon2 = np.radians(flr_lat), np.radians(flr_lon)
        
        cos_d = np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lon1 - lon2)
        dist_deg = np.degrees(np.arccos(np.clip(cos_d, -1.0, 1.0)))
        
        # Scoring scale
        if dist_deg <= 10.0:
            return 1.0
        elif dist_deg <= 20.0:
            return 0.7
        elif dist_deg <= 30.0:
            return 0.4
        elif dist_deg <= 45.0:
            return 0.1
        else:
            return 0.0

    def compute_active_region_score(self, fil_ar: Any, flr_ar: Any) -> float:
        """Score based on active region numbers."""
        if pd.isna(fil_ar) or pd.isna(flr_ar) or fil_ar is None or flr_ar is None:
            return 0.5  # Neutral if missing
            
        try:
            # Coerce to ints for comparison
            ar1 = int(float(fil_ar))
            ar2 = int(float(flr_ar))
            return 1.0 if ar1 == ar2 else 0.0
        except ValueError:
            # String comparison
            s1 = str(fil_ar).strip()
            s2 = str(flr_ar).strip()
            return 1.0 if s1 == s2 else 0.0

    def compute_association_score(self, filament: Dict[str, Any], flare: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """Compute the weighted association score and return individual component scores."""
        # Get filament features
        fil_time = filament.get("timestamp") or filament.get("image_timestamp")
        fil_lat = filament.get("solar_lat", float("nan"))
        fil_lon = filament.get("solar_lon", float("nan"))
        fil_ar = filament.get("active_region")
        
        # Get flare features
        flr_time = flare.get("start_time")
        flr_loc = flare.get("source_location")
        flr_ar = flare.get("active_region")
        
        # Calculate individual scores
        s_temp = self.compute_temporal_score(fil_time, flr_time)
        s_spat = self.compute_spatial_score(fil_lat, fil_lon, flr_loc)
        s_ar = self.compute_active_region_score(fil_ar, flr_ar)
        
        # Use placeholders for eruption and direction if not provided
        s_erupt = 1.0 if filament.get("eruption_indicator") else 0.5
        s_dir = 1.0 if filament.get("directional_consistency") else 0.5
        
        scores = {
            "active_region_match": s_ar,
            "temporal_score": s_temp,
            "spatial_score": s_spat,
            "eruption_indicator": s_erupt,
            "directional_consistency": s_dir
        }
        
        # Weighted sum. If spatial score is NaN, redistribute its weight to temporal and active region.
        w = self.weights.copy()
        if np.isnan(s_spat):
            # Spatial score is NaN: drop spatial and use default weights normalized
            # Let's redistribute the spatial weight of 0.20:
            # 0.12 to active_region_match (now 0.52), 0.08 to temporal_score (now 0.33)
            w["active_region_match"] += 0.12
            w["temporal_score"] += 0.08
            w["spatial_score"] = 0.0
            s_spat_val = 0.0
        else:
            s_spat_val = s_spat
            
        score = (
            w["active_region_match"] * s_ar
            + w["temporal_score"] * s_temp
            + w["spatial_score"] * s_spat_val
            + w["eruption_indicator"] * s_erupt
            + w["directional_consistency"] * s_dir
        )
        
        return float(score), scores

    def classify_association(self, score: float, s_spat: float) -> str:
        """Classify into confidence bands."""
        if score >= self.high_thresh:
            return "HIGH_CONFIDENCE_ASSOCIATION"
        elif score >= self.med_thresh:
            return "MEDIUM_CONFIDENCE_ASSOCIATION"
        elif score >= self.low_thresh:
            return "LOW_CONFIDENCE_ASSOCIATION"
        else:
            return "UNMATCHED"

    def associate(self, filaments: List[Dict[str, Any]], flares: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Link filaments to candidates and save to CSV."""
        links = []
        
        for fil in filaments:
            fil_id = fil.get("filament_id", "unknown")
            fil_time = fil.get("timestamp") or fil.get("image_timestamp")
            
            best_flare = None
            best_score = -1.0
            best_components = {}
            
            for flr in flares:
                score, components = self.compute_association_score(fil, flr)
                if score > best_score:
                    best_score = score
                    best_flare = flr
                    best_components = components
                    
            # Classify best match
            s_spat = best_components.get("spatial_score", float("nan"))
            label = self.classify_association(best_score, s_spat)
            
            # Record association details
            record = {
                "filament_id": fil_id,
                "observation_time": fil_time,
                "flare_id": best_flare.get("flare_id") if best_flare else "N/A",
                "flare_start": best_flare.get("start_time") if best_flare else "N/A",
                "flare_peak": best_flare.get("peak_time") if best_flare else "N/A",
                "flare_class": best_flare.get("class_type") if best_flare else "N/A",
                "active_region_match": best_components.get("active_region_match", 0.5),
                "temporal_score": best_components.get("temporal_score", 0.0),
                "spatial_score": best_components.get("spatial_score", float("nan")),
                "eruption_indicator": best_components.get("eruption_indicator", 0.5),
                "directional_score": best_components.get("directional_consistency", 0.5),
                "association_score": round(best_score, 3) if best_flare else 0.0,
                "association_label": label if best_flare else "UNMATCHED"
            }
            links.append(record)
            
        self._save_links(links)
        return links

    def _save_links(self, links: List[Dict[str, Any]]):
        if not links:
            return
        df = pd.DataFrame(links)
        if self.links_csv.exists():
            try:
                old_df = pd.read_csv(self.links_csv)
                combined = pd.concat([old_df, df], ignore_index=True)
                # Deduplicate by filament_id + observation_time
                combined = combined.drop_duplicates(subset=["filament_id", "observation_time"], keep="last")
                combined.to_csv(self.links_csv, index=False)
            except Exception as e:
                logger.error(f"Error updating links CSV: {e}")
                df.to_csv(self.links_csv, index=False)
        else:
            df.to_csv(self.links_csv, index=False)
            logger.info(f"Created new links CSV file: {self.links_csv}")


def generate_training_table(filaments: List[Dict[str, Any]], links: List[Dict[str, Any]],
                            output_csv: str = "data/training/filament_flare_training.csv") -> pd.DataFrame:
    """Generate training table from filaments and their associated flare links."""
    # Build a lookup for links
    link_lookup = {(str(l["filament_id"]), str(l["observation_time"])): l for l in links}
    
    rows = []
    for fil in filaments:
        fil_id = str(fil.get("filament_id"))
        fil_time = fil.get("timestamp") or fil.get("image_timestamp")
        
        # Default target values
        target_m_or_x = 0
        target_c_or_higher = 0
        flare_class = None
        
        link = link_lookup.get((fil_id, fil_time))
        if link and link.get("association_label") in ["HIGH_CONFIDENCE", "MEDIUM_CONFIDENCE"]:
            fl_class = link.get("flare_class")
            if fl_class and isinstance(fl_class, str) and fl_class != "N/A":
                flare_class = fl_class
                fl_class = fl_class.upper()
                if fl_class.startswith(("M", "X")):
                    target_m_or_x = 1
                if fl_class.startswith(("C", "M", "X")):
                    target_c_or_higher = 1
                    
        # Static morphology features
        row = {
            "filament_id": fil_id,
            "observation_time": fil_time,
            "area_px": fil.get("area_px", float("nan")),
            "perimeter_px": fil.get("perimeter_px", float("nan")),
            "skeleton_length_px": fil.get("skeleton_length_px", float("nan")),
            "avg_width_px": fil.get("avg_width_px", float("nan")),
            "aspect_ratio": fil.get("aspect_ratio", float("nan")),
            "sinuosity": fil.get("sinuosity", 1.0),
            "compactness": fil.get("compactness", float("nan")),
            "confidence": fil.get("confidence", 0.0),
            
            # Solar context
            "disk_center_dist": fil.get("disk_center_dist", float("nan")),
            "solar_lat": fil.get("solar_lat", float("nan")),
            "solar_lon": fil.get("solar_lon", float("nan")),
            "active_region": fil.get("active_region"),
            "eruption_indicator": fil.get("eruption_indicator"),
            
            # Temporal features (defaults to nan initially)
            "area_growth_rate": fil.get("area_growth_rate", float("nan")),
            "length_growth_rate": fil.get("length_growth_rate", float("nan")),
            "width_growth_rate": fil.get("width_growth_rate", float("nan")),
            "centroid_velocity": fil.get("centroid_velocity", float("nan")),
            "orientation_change": fil.get("orientation_change", float("nan")),
            
            # Historical context
            "flare_count_prev_24h": fil.get("flare_count_prev_24h", 0),
            "max_recent_flare_class": fil.get("max_recent_flare_class", "None"),
            
            # Targets
            "target_M_or_X": target_m_or_x,
            "target_C_or_higher": target_c_or_higher,
            "flare_class": flare_class
        }
        rows.append(row)
        
    df = pd.DataFrame(rows)
    if not df.empty:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        # Deduplicate
        if os.path.exists(output_csv):
            try:
                old_df = pd.read_csv(output_csv)
                combined = pd.concat([old_df, df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["filament_id", "observation_time"], keep="last")
                combined.to_csv(output_csv, index=False)
                df = combined
            except Exception as e:
                df.to_csv(output_csv, index=False)
        else:
            df.to_csv(output_csv, index=False)
            
    return df

