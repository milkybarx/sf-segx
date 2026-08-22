import math
import unittest
import numpy as np
import pickle
import json
from pathlib import Path
import pandas as pd
from analysis.coordinates import parse_stonyhurst_string, pixel_to_stonyhurst, stonyhurst_to_pixel
from analysis.spacecraft_catalog import SpacecraftCatalog
from analysis.association import FlareAssociationEngine

class TestCoordinatesAndExposure(unittest.TestCase):
    
    def test_stonyhurst_string_parsing(self):
        """Verify parsing of Stonyhurst coordinate notation strings into lat/lon float degrees."""
        # North-West
        lat, lon = parse_stonyhurst_string("N18W45")
        self.assertEqual(lat, 18.0)
        self.assertEqual(lon, 45.0)
        
        # South-East
        lat, lon = parse_stonyhurst_string("S20E10")
        self.assertEqual(lat, -20.0)
        self.assertEqual(lon, -10.0)
        
        # Boundary / missing fields
        lat, lon = parse_stonyhurst_string("N15")
        self.assertEqual(lat, 15.0)
        self.assertIsNone(lon)
        
        lat, lon = parse_stonyhurst_string("W30")
        self.assertIsNone(lat)
        self.assertEqual(lon, 30.0)
        
        # Malformed strings
        lat, lon = parse_stonyhurst_string("invalid")
        self.assertIsNone(lat)
        self.assertIsNone(lon)
        
        lat, lon = parse_stonyhurst_string(None)
        self.assertIsNone(lat)
        self.assertIsNone(lon)

    def test_pixel_to_stonyhurst_conversion(self):
        """Test converting pixel coordinates relative to solar disk center to Stonyhurst coordinates."""
        cx, cy, R = 256.0, 256.0, 200.0
        
        # Center of disk (0, 0)
        lat, lon = pixel_to_stonyhurst(cx, cy, cx, cy, R)
        self.assertEqual(lat, 0.0)
        self.assertEqual(lon, 0.0)
        
        # North pole (Y = cy - R, X = cx)
        lat, lon = pixel_to_stonyhurst(cx, cy - R, cx, cy, R)
        self.assertAlmostEqual(lat, 90.0, places=4)
        self.assertAlmostEqual(lon, 0.0, places=4)
        
        # South pole (Y = cy + R, X = cx)
        lat, lon = pixel_to_stonyhurst(cx, cy + R, cx, cy, R)
        self.assertAlmostEqual(lat, -90.0, places=4)
        self.assertAlmostEqual(lon, 0.0, places=4)
        
        # West limb (X = cx + R, Y = cy) -> West is positive longitude
        lat, lon = pixel_to_stonyhurst(cx + R, cy, cx, cy, R)
        self.assertAlmostEqual(lat, 0.0, places=4)
        self.assertAlmostEqual(lon, 90.0, places=4)
        
        # East limb (X = cx - R, Y = cy) -> East is negative longitude
        lat, lon = pixel_to_stonyhurst(cx - R, cy, cx, cy, R)
        self.assertAlmostEqual(lat, 0.0, places=4)
        self.assertAlmostEqual(lon, -90.0, places=4)
        
        # Off limb boundary check
        lat, lon = pixel_to_stonyhurst(cx + R + 5.0, cy, cx, cy, R)
        self.assertTrue(np.isnan(lat))
        self.assertTrue(np.isnan(lon))

    def test_stonyhurst_to_pixel_conversion(self):
        """Check mapping heliographic coordinates back to 2D image coordinates."""
        cx, cy, R = 256.0, 256.0, 200.0
        
        # Center
        x, y = stonyhurst_to_pixel(0.0, 0.0, cx, cy, R)
        self.assertEqual(x, cx)
        self.assertEqual(y, cy)
        
        # North Pole
        x, y = stonyhurst_to_pixel(90.0, 0.0, cx, cy, R)
        self.assertAlmostEqual(x, cx, places=4)
        self.assertAlmostEqual(y, cy - R, places=4)

    def test_spacecraft_cme_exposure(self):
        """Test geometric cone exposure models for spacecraft positions."""
        catalog = SpacecraftCatalog()
        
        # SOHO is at L1 (lat=0, lon=0, dist=0.99 AU)
        # CME directed at Earth: lat=0, lon=0, half-angle=45, speed=1000 km/s
        res = catalog.calculate_cme_exposure("SOHO", 0.0, 0.0, 45.0, 1000.0, "2026-08-22 12:00")
        self.assertEqual(res["exposure_type"], "INSIDE_CONE")
        self.assertTrue(res["travel_time_hours"] > 0)
        
        # Glancing CME: lat=0, lon=50, half-angle=42 (angular separation is 50, which is within 50 - 42 = 8 of flank)
        res_glancing = catalog.calculate_cme_exposure("SOHO", 0.0, 50.0, 42.0, 1000.0, "2026-08-22 12:00")
        self.assertEqual(res_glancing["exposure_type"], "NEAR_FLANK")
        
        # Miss CME: lat=0, lon=70, half-angle=30 (separation 70, outside 30+10 flank)
        res_miss = catalog.calculate_cme_exposure("SOHO", 0.0, 70.0, 30.0, 1000.0, "2026-08-22 12:00")
        self.assertEqual(res_miss["exposure_type"], "OUTSIDE")

    def test_association_engine(self):
        """Verify temporal and spatial scores from association engine."""
        engine = FlareAssociationEngine()
        
        # Same Active Region, close time
        fil = {"filament_id": "1", "timestamp": "2026-08-22T12:00:00", "solar_lat": 10.0, "solar_lon": 15.0, "active_region": 13456}
        flr = {"flare_id": "F1", "start_time": "2026-08-22T13:30:00", "source_location": "N10W15", "active_region": 13456}
        
        score, components = engine.compute_association_score(fil, flr)
        self.assertEqual(components["active_region_match"], 1.0)
        self.assertEqual(components["temporal_score"], 1.0)  # +1.5h difference is optimal
        self.assertEqual(components["spatial_score"], 1.0)  # exact spatial coordinates
        self.assertTrue(score >= 0.8)
        self.assertEqual(engine.classify_association(score, 1.0), "HIGH_CONFIDENCE_ASSOCIATION")
        
        # Extreme temporal offset
        flr_late = {"flare_id": "F2", "start_time": "2026-08-23T15:00:00", "source_location": "N10W15", "active_region": 13456}
        _, comp_late = engine.compute_association_score(fil, flr_late)
        self.assertEqual(comp_late["temporal_score"], 0.0)

    def test_forecast_dataset_structure(self):
        """Verify structural properties of the refined forecasting dataset."""
        csv_path = Path("data/training/filament_forecast_training.csv")
        self.assertTrue(csv_path.exists())
        
        df = pd.read_csv(csv_path)
        self.assertFalse(df.empty)
        
        # Check uniqueness
        dup_keys = df.duplicated(subset=["filament_id", "observation_time"])
        self.assertEqual(dup_keys.sum(), 0)
        
        # Check required columns
        required_cols = [
            "filament_id", "observation_time", "image_id", "dataset_source", "subset", "split",
            "M_X_WITHIN_24H", "M_OR_HIGHER_24H", "X_CLASS_24H", "recent_flare_count"
        ]
        for col in required_cols:
            self.assertIn(col, df.columns)
            
        # Check chronological split boundaries
        train_max = pd.to_datetime(df[df["split"] == "TRAIN"]["observation_time"]).max()
        val_min = pd.to_datetime(df[df["split"] == "VAL"]["observation_time"]).min()
        test_min = pd.to_datetime(df[df["split"] == "TEST"]["observation_time"]).min()
        
        if not pd.isna(train_max) and not pd.isna(val_min):
            self.assertTrue(train_max <= val_min)
        if not pd.isna(val_min) and not pd.isna(test_min):
            self.assertTrue(val_min <= test_min)

    def test_risk_models_existence_and_loading(self):
        """Verify that model pickles are saved and can be successfully unpickled."""
        model_path = Path("experiments/phase2c_flare_risk/best_flare_risk_model.pkl")
        self.assertTrue(model_path.exists())
        
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        self.assertIsNotNone(model)
        
        # Verify metadata loading
        meta_path = Path("experiments/phase2c_flare_risk/best_flare_risk_metadata.json")
        self.assertTrue(meta_path.exists())
        with open(meta_path, "r") as f:
            meta = json.load(f)
        self.assertEqual(meta["dataset_version"], "MAGFiLO_40_gallery_validation")


if __name__ == "__main__":
    unittest.main()
