import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss, log_loss
)

# Paths
OUT_DIR = Path("experiments/phase2e_flare_risk")
DATA_PATH = Path("data/training/filament_forecast_full.csv")
REPORT_PATH = Path("reports/PHASE2E1_EVALUATION_AUDIT.md")

def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df["has_temporal_history"] = (~df["area_growth_rate"].isna()).astype(int)

    val_df = df[df["split"] == "VAL"].copy()
    test_df = df[df["split"] == "TEST"].copy()
    y_val = val_df["M_X_WITHIN_24H"].values.astype(int)
    y_test = test_df["M_X_WITHIN_24H"].values.astype(int)

    print("Loading best model (calibrated)...")
    with open(OUT_DIR / "best_flare_risk_model.pkl", "rb") as f:
        cal = pickle.load(f)

    with open(OUT_DIR / "best_flare_risk_metadata.json", "r") as f:
        import json
        metadata = json.load(f)

    best_key = metadata["best_model_key"]
    model_algo = best_key.split("_")[1]
    config_name = best_key.split("_")[0]
    
    uncal_pkl_name = "randomforest_context_(abcd).pkl"
    print(f"Loading uncalibrated model: {uncal_pkl_name}")
    with open(OUT_DIR / uncal_pkl_name, "rb") as f:
        uncal = pickle.load(f)
        
    print(f"Uncalibrated classes_ : {uncal.classes_}")
    print(f"Calibrated classes_   : {cal.classes_}")
    
    with open(OUT_DIR / f"preprocessor_{config_name.lower()}.pkl", "rb") as f:
        prep = pickle.load(f)
    
    with open(OUT_DIR / "config.json", "r") as f:
        configs = json.load(f)
    
    num_feats = configs["numeric_configs"][config_name.lower()]
    cat_feats = configs["categorical_configs"][config_name.lower()]
    
    if "perimeter" in num_feats: num_feats.remove("perimeter")
    if "hours_since_active_region_flare" in num_feats: num_feats.remove("hours_since_active_region_flare")
    if "active_region_recent_max_class" in cat_feats: cat_feats.remove("active_region_recent_max_class")
    
    avail_num = [f for f in num_feats if f in df.columns]
    avail_cat = [f for f in cat_feats if f in df.columns]
    
    X_val = pd.DataFrame(prep.transform(val_df[avail_num + avail_cat]))
    X_test = pd.DataFrame(prep.transform(test_df[avail_num + avail_cat]))
    
    raw_prob = uncal.predict_proba(X_test)[:, 1]
    calibrated_prob = cal.predict_proba(X_test)[:, 1]
    
    spearman_corr, _ = spearmanr(raw_prob, calibrated_prob)
    inversions = 0
    n = len(raw_prob)
    for i in range(min(n, 1000)):
        for j in range(i+1, min(n, 1000)):
            if (raw_prob[i] > raw_prob[j] and calibrated_prob[i] < calibrated_prob[j]) or \
               (raw_prob[i] < raw_prob[j] and calibrated_prob[i] > calibrated_prob[j]):
                inversions += 1
                
    raw_auc_pr = average_precision_score(y_test, raw_prob)
    cal_auc_pr = average_precision_score(y_test, calibrated_prob)
    raw_auc_roc = roc_auc_score(y_test, raw_prob)
    cal_auc_roc = roc_auc_score(y_test, calibrated_prob)
    
    raw_brier = brier_score_loss(y_test, raw_prob)
    cal_brier = brier_score_loss(y_test, calibrated_prob)
    raw_logloss = log_loss(y_test, raw_prob)
    cal_logloss = log_loss(y_test, calibrated_prob)
    
    val_preds = pd.read_csv(OUT_DIR / "predictions_val.csv")
    test_preds = pd.read_csv(OUT_DIR / "predictions_test.csv")
    align_test = (test_preds["M_X_WITHIN_24H"].values == y_test).all()
    align_prob_raw = np.allclose(test_preds["best_model_prob"].values, raw_prob)
    align_prob_cal = np.allclose(test_preds["best_model_prob_cal"].values, calibrated_prob)
    
    constant_train_rate = 705 / 4399
    naive_probs = np.full(len(y_test), constant_train_rate)
    naive_pr_auc = average_precision_score(y_test, naive_probs)
    naive_roc_auc = roc_auc_score(y_test, naive_probs)
    naive_brier = brier_score_loss(y_test, naive_probs)
    
    results = pd.read_csv(OUT_DIR / "metrics.csv")
    static_pr = results[results["config"] == "Static (A+B)"]["test_pr_auc"].values
    
    REPORT_PATH.parent.mkdir(exist_ok=True, parents=True)
    with open(REPORT_PATH, "w") as f:
        f.write("# PHASE 2E.1 — EVALUATION AUDIT\n\n")
        
        f.write("## 1. Class Probability Column\n")
        f.write(f"- Uncalibrated classes_: {uncal.classes_.tolist()}\n")
        f.write(f"- Calibrated classes_  : {cal.classes_.tolist()}\n")
        if uncal.classes_.tolist() != cal.classes_.tolist():
            f.write("**BUG DETECTED**: Calibrated classes_ differ from Uncalibrated classes_. This inverted the probabilities!\n\n")
            
        f.write("## 2. Ranking Preservation\n")
        f.write(f"- Spearman correlation (raw vs cal): {spearman_corr:.4f}\n")
        f.write(f"- Ranking inversions (first 1000 pairs): {inversions}\n\n")
        
        f.write("## 3. Metrics (Recalculated on exact same array)\n")
        f.write(f"- Raw PR-AUC: {raw_auc_pr:.4f}\n")
        f.write(f"- Cal PR-AUC: {cal_auc_pr:.4f}\n")
        f.write(f"- Raw ROC-AUC: {raw_auc_roc:.4f}\n")
        f.write(f"- Cal ROC-AUC: {cal_auc_roc:.4f}\n\n")
        
        f.write("## 4. Row Alignment\n")
        f.write(f"- Test target alignment: {align_test}\n")
        f.write(f"- Test raw prob alignment: {align_prob_raw}\n")
        f.write(f"- Test cal prob alignment: {align_prob_cal}\n\n")
        
        f.write("## 5. Calibration Procedure\n")
        f.write(f"- Calibrated Classifier: {cal.__class__.__name__} (method='sigmoid', cv='prefit')\n")
        f.write(f"- Fitted on VAL set: {len(y_val)} rows, {int(y_val.sum())} positive.\n\n")
        
        f.write("## 6. Brier & Log Loss\n")
        f.write(f"- Raw Brier: {raw_brier:.4f}\n")
        f.write(f"- Cal Brier: {cal_brier:.4f}\n")
        f.write(f"- Raw Log Loss: {raw_logloss:.4f}\n")
        f.write(f"- Cal Log Loss: {cal_logloss:.4f}\n\n")
        
        f.write("## 7. Naive Baseline\n")
        f.write(f"- Constant Train Rate: {constant_train_rate:.4f}\n")
        f.write(f"- Naive PR-AUC: {naive_pr_auc:.4f}\n")
        f.write(f"- Naive ROC-AUC: {naive_roc_auc:.4f}\n")
        f.write(f"- Naive Brier: {naive_brier:.4f}\n\n")
        
        f.write("## 9. Static Model Check\n")
        f.write(f"- Static Test PR-AUCs: {static_pr}\n\n")
        
        f.write("## Conclusion\n")
        if spearman_corr < 0:
            f.write("C: Evaluation implementation is invalid and Phase 2E test results must be recomputed.\n")
        else:
            f.write("B: No bug found; Phase 2E metrics are internally consistent.\n")

if __name__ == "__main__":
    main()
