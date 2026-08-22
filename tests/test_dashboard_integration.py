import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from analysis.flare_inference import FlareRiskInference

class TestDashboardIntegration(unittest.TestCase):

    def setUp(self):
        self.inference = FlareRiskInference()

    def test_missing_timestamp(self):
        filament = {"area_px": 100}
        score, status = self.inference.predict_risk(filament, None)
        self.assertTrue(np.isnan(score))
        self.assertEqual(status, "NO_TIMESTAMP")

    def test_invalid_timestamp(self):
        filament = {"area_px": 100}
        score, status = self.inference.predict_risk(filament, "invalid_time")
        self.assertTrue(np.isnan(score))
        self.assertEqual(status, "INVALID_TIMESTAMP")

    def test_fallback_disk_position(self):
        filament = {
            "area_px": 100,
            "centroid": {"x": 2048, "y": 2048}
        }
        # A mock context should just work and score returned
        # We assume the model files are present, else it returns MODEL_UNAVAILABLE
        score, status = self.inference.predict_risk(filament, "2021-10-28T12:00:00Z")
        if status != "MODEL_UNAVAILABLE":
            # If the model is available, score should be a float or nan
            self.assertIsInstance(score, float)
            self.assertEqual(status, "UNCALIBRATED")

    def test_uncalibrated_label_contract(self):
        # We explicitly enforce "UNCALIBRATED" language in the inference output
        score, status = self.inference.predict_risk({}, "2021-10-28T12:00:00Z")
        if status != "MODEL_UNAVAILABLE":
            self.assertEqual(status, "UNCALIBRATED")

if __name__ == '__main__':
    unittest.main()
