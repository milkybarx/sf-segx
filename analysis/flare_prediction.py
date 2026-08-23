import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import numpy as np
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Cache the model so we don't load it on every function call
_rf_model = None

def get_rf_model():
    global _rf_model
    if _rf_model is None:
        model_path = os.path.join(BASE_DIR, "checkpoints", "flare_rf_model.pkl")
        if os.path.exists(model_path):
            _rf_model = joblib.load(model_path)
    return _rf_model

def calculate_flare_probability(length_px: float, region_type: str) -> float:
    """
    Computes a solar flare eruption probability score using a trained Random Forest ML model
    based on historical flare data.
    """
    model = get_rf_model()
    
    # If model hasn't been trained yet, fallback to heuristic
    if model is None:
        if region_type == "ARF": baseline = 0.45
        elif region_type == "IRF": baseline = 0.20
        else: baseline = 0.05
        length_factor = 1.0 - np.exp(-0.005 * length_px)
        prob = baseline + (1.0 - baseline) * length_factor
        return float(np.clip(prob, 0.0, 0.98))
        
    # Map categorical Region Type for ML inference
    region_map = {"QRF": 0, "IRF": 1, "ARF": 2}
    region_code = region_map.get(region_type, 0)
    
    # Random Forest expects a 2D array: [n_samples, n_features]
    # Features must match training: ['length_px', 'region_code']
    # We must construct a DataFrame or use an array if no feature names are enforced.
    # scikit-learn models handle arrays perfectly if they were trained on pandas with matching columns.
    features = np.array([[length_px, region_code]])
    
    # Predict probability of class 1 (Flare)
    probs = model.predict_proba(features)
    # If model only has one class (0), probs might only have 1 column. Handle gracefully:
    if probs.shape[1] > 1:
        flare_prob = float(probs[0, 1])
    else:
        flare_prob = 0.0
    
    return flare_prob

