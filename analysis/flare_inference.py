import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple
from analysis.donki.donki_client import DONKIClient

MODEL_PATH = "experiments/phase2e_flare_risk/best_flare_risk_model.pkl"
PREPROCESSOR_PATH = "experiments/phase2e_flare_risk/preprocessor_context.pkl"

class FlareRiskInference:
    """Wrapper for Phase 2E.2 RandomForest + Context model."""
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self._load_models()
        self.donki = DONKIClient()
        
    def _load_models(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
            self.model = joblib.load(MODEL_PATH)
            self.preprocessor = joblib.load(PREPROCESSOR_PATH)
            
    def _compute_context(self, obs_time: datetime) -> Dict[str, Any]:
        """Fetch DONKI context strictly PRIOR to obs_time to match Phase 2E."""
        # Query up to 7 days prior
        start_str = (obs_time - timedelta(days=7)).strftime("%Y-%m-%d")
        end_str = obs_time.strftime("%Y-%m-%d")
        
        try:
            flares = self.donki.get_flares(start_str, end_str)
        except Exception:
            flares = []
            
        # Filter strictly prior to obs_time
        valid_flares = []
        for f in flares:
            st = f.get("start_time") or f.get("peak_time")
            if not st: continue
            try:
                dt = datetime.fromisoformat(st.replace("Z", ""))
                if dt < obs_time:
                    valid_flares.append({"dt": dt, "class": f.get("class_type", "A")})
            except Exception:
                pass
                
        # Compute metrics
        now24h = obs_time - timedelta(hours=24)
        recent_flares = [f for f in valid_flares if f["dt"] >= now24h]
        
        c_count = sum(1 for f in recent_flares if f["class"].startswith("C"))
        m_count = sum(1 for f in recent_flares if f["class"].startswith("M"))
        x_count = sum(1 for f in recent_flares if f["class"].startswith("X"))
        
        if valid_flares:
            last_flare = max(valid_flares, key=lambda x: x["dt"])
            hrs_since = (obs_time - last_flare["dt"]).total_seconds() / 3600.0
        else:
            hrs_since = np.nan
            
        classes = [f["class"] for f in recent_flares]
        max_class = "NONE"
        if classes:
            if any(c.startswith("X") for c in classes): max_class = "X"
            elif any(c.startswith("M") for c in classes): max_class = "M"
            elif any(c.startswith("C") for c in classes): max_class = "C"
            else: max_class = "B"
            
        return {
            "recent_flare_count": len(recent_flares) if valid_flares else np.nan,
            "recent_C_count": c_count if valid_flares else np.nan,
            "recent_M_count": m_count if valid_flares else np.nan,
            "recent_X_count": x_count if valid_flares else np.nan,
            "hours_since_previous_flare": hrs_since,
            "recent_max_flare_class": max_class if valid_flares else "NONE",
            # Phase 2E schema dropped AR tracking for simplicity in dashboard, put NaNs 
            "active_region_previous_flare_count": np.nan,
            "active_region_previous_M_count": np.nan,
            "active_region_previous_X_count": np.nan
        }
        
    def predict_risk(self, filament: Dict[str, Any], timestamp_str: str = None) -> Tuple[float, str]:
        """Predict relative flare risk. Returns (score, status)."""
        if not self.model or not self.preprocessor:
            return float('nan'), "MODEL_UNAVAILABLE"
            
        if not timestamp_str:
            return float('nan'), "NO_TIMESTAMP"
            
        try:
            # Clean ISO string if it ends in Z
            clean_ts = timestamp_str.replace("Z", "")
            if len(clean_ts) == 8: # YYYYMMDD
                obs_time = datetime.strptime(clean_ts, "%Y%m%d")
            else:
                obs_time = datetime.fromisoformat(clean_ts)
        except Exception:
            return float('nan'), "INVALID_TIMESTAMP"
            
        # 1. Fetch historical context
        ctx = self._compute_context(obs_time)
        
        # 2. Extract morphology (Phase 2E features)
        df = pd.DataFrame([{
            "area": filament.get("area_px", np.nan),
            "length": filament.get("skeleton_length_px", np.nan),
            "width": filament.get("avg_width_px", np.nan),
            "skeleton_length": filament.get("skeleton_length_px", np.nan),
            "aspect_ratio": filament.get("aspect_ratio", np.nan),
            "sinuosity": filament.get("sinuosity", np.nan),
            "orientation": filament.get("orientation_deg", np.nan),
            "confidence": filament.get("confidence", np.nan),
            "centroid_lat": filament.get("centroid", {}).get("y", np.nan),
            "centroid_lon": filament.get("centroid", {}).get("x", np.nan),
            "disk_position": filament.get("disk_position", np.nan), # Typically normalized distance from center
            "filament_type": filament.get("filament_type", "UNKNOWN"),
            "filament_rating": filament.get("filament_rating", "NONE"),
            
            # Context
            "recent_flare_count": ctx["recent_flare_count"],
            "recent_C_count": ctx["recent_C_count"],
            "recent_M_count": ctx["recent_M_count"],
            "recent_X_count": ctx["recent_X_count"],
            "hours_since_previous_flare": ctx["hours_since_previous_flare"],
            "active_region_previous_flare_count": ctx["active_region_previous_flare_count"],
            "active_region_previous_M_count": ctx["active_region_previous_M_count"],
            "active_region_previous_X_count": ctx["active_region_previous_X_count"],
            "recent_max_flare_class": ctx["recent_max_flare_class"]
        }])
        
        # We need disk_position fallback if not provided
        if pd.isna(df.iloc[0]["disk_position"]):
            # Normalize centroid (assume roughly 4096x4096 center is 2048)
            cx, cy = df.iloc[0]["centroid_lon"], df.iloc[0]["centroid_lat"]
            if not pd.isna(cx) and not pd.isna(cy):
                # Distance from center, scaled to roughly 0-1
                dist = np.sqrt((cx - 2048)**2 + (cy - 2048)**2)
                df.at[0, "disk_position"] = dist / 2048.0
                
        # 3. Transform and predict
        try:
            X = self.preprocessor.transform(df)
            probs = self.model.predict_proba(X)
            # Find positive class index (usually 1, but best to check classes_)
            pos_idx = list(self.model.classes_).index(1) if 1 in self.model.classes_ else 1
            score = float(probs[0, pos_idx])
            return score, "UNCALIBRATED"
        except Exception as e:
            print(f"Inference error: {e}")
            return float('nan'), f"ERROR: {e}"
