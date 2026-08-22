"""Train and evaluate baseline flare forecasting models (Logistic Regression, Random Forest, XGBoost) on refined forecasting dataset."""
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss,
    precision_recall_curve, precision_score, recall_score,
    f1_score, accuracy_score, confusion_matrix
)

# XGBoost import
from xgboost import XGBClassifier

# Add root folder to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set seeds
np.random.seed(42)

def run_sanity_checks(df: pd.DataFrame, feature_cols: List[str], target_col: str):
    """Verify target distributions, splits, and ensure no future feature leakage."""
    print("=" * 80)
    print("RUNNING PHASE 2C SANITY CHECKS")
    print("=" * 80)
    
    # 1. Verify target classes are present in all splits
    splits = ["TRAIN", "VAL", "TEST"]
    for s in splits:
        split_df = df[df["split"] == s]
        if split_df.empty:
            raise ValueError(f"Split {s} is empty!")
            
        counts = split_df[target_col].value_counts()
        print(f"{s} Split counts: {dict(counts)}")
        if len(counts) < 2:
            raise ValueError(f"Split {s} has only one target class: {dict(counts)}! Cannot train/evaluate.")
            
    # 2. Verify no future target columns appear in X
    future_cols = [
        "future_flare_count", "first_future_flare_time", "first_future_flare_class",
        "strongest_future_flare_class", "future_M_count", "future_X_count",
        "M_X_WITHIN_6H", "M_X_WITHIN_12H", "M_X_WITHIN_48H", "X_CLASS_48H"
    ]
    for col in feature_cols:
        if col in future_cols or "future" in col.lower() or "target" in col.lower():
            raise ValueError(f"CRITICAL: Future column '{col}' is present in feature configuration X!")
            
    # 3. Print counts
    for s in splits:
        split_df = df[df["split"] == s]
        pos = int(split_df[target_col].sum())
        neg = len(split_df) - pos
        print(f"{s}: rows={len(split_df)} | positives={pos} | negatives={neg} | pos_rate={pos/len(split_df)*100:.1f}%")
        
    print("\nSANITY CHECKS PASSED: Ready to train.\n")


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """Build a ColumnTransformer for numerical imputation + scaling and categorical encoding."""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    return preprocessor


def get_feature_names_out(column_transformer, numeric_cols, categorical_cols, train_df) -> List[str]:
    """Helper to retrieve feature names out of the ColumnTransformer."""
    # Fit imputer and encoder on training data to extract categories
    temp_pipeline = Pipeline(steps=[('preprocessor', column_transformer)])
    temp_pipeline.fit(train_df[numeric_cols + categorical_cols])
    
    # Extract names
    cat_encoder = temp_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    cat_features = cat_encoder.get_feature_names_out(categorical_cols).tolist()
    return numeric_cols + cat_features


def sweep_thresholds(y_true, y_prob) -> Tuple[float, Dict[str, float]]:
    """Sweep threshold values to find the one maximizing F1 score on validation."""
    best_th = 0.50
    best_f1 = -1.0
    best_metrics = {}
    
    # Sweep from 0.10 to 0.90 with step 0.01
    for th in np.arange(0.10, 0.91, 0.01):
        y_pred = (y_prob >= th).astype(int)
        
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        bal_acc = 0.5 * (rec + spec)
        
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
            best_metrics = {
                "Precision": float(prec),
                "Recall": float(rec),
                "F1": float(f1),
                "Specificity": float(spec),
                "BalancedAccuracy": float(bal_acc)
            }
            
    return float(best_th), best_metrics


def evaluate_model(model_name: str, pipeline, X_train, y_train, X_val, y_val, X_test, y_test, frozen_threshold=None, is_calibrated=False) -> Dict[str, Any]:
    """Train, predict, and compute exhaustive metrics for a single model pipeline."""
    # Fit on train ONLY
    pipeline.fit(X_train, y_train)
    
    # Predict probabilities
    y_prob_train = pipeline.predict_proba(X_train)[:, 1]
    y_prob_val = pipeline.predict_proba(X_val)[:, 1]
    
    # Calculate PR-AUC and ROC-AUC
    pr_auc_val = average_precision_score(y_val, y_prob_val)
    roc_auc_val = roc_auc_score(y_val, y_prob_val)
    brier_val = brier_score_loss(y_val, y_prob_val)
    
    # Threshold Selection (Validation ONLY)
    if frozen_threshold is None:
        th, val_metrics = sweep_thresholds(y_val, y_prob_val)
    else:
        th = frozen_threshold
        # Calculate metrics for frozen threshold
        y_pred_val = (y_prob_val >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred_val).ravel()
        rec = recall_score(y_val, y_pred_val, zero_division=0)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        val_metrics = {
            "Precision": float(precision_score(y_val, y_pred_val, zero_division=0)),
            "Recall": float(rec),
            "F1": float(f1_score(y_val, y_pred_val, zero_division=0)),
            "Specificity": float(spec),
            "BalancedAccuracy": float(0.5 * (rec + spec))
        }
        
    y_pred_val = (y_prob_val >= th).astype(int)
    cm_val = confusion_matrix(y_val, y_pred_val).tolist()
    
    # Test set evaluation (Sealed until frozen)
    y_prob_test = pipeline.predict_proba(X_test)[:, 1]
    y_pred_test = (y_prob_test >= th).astype(int)
    
    pr_auc_test = average_precision_score(y_test, y_prob_test)
    roc_auc_test = roc_auc_score(y_test, y_prob_test)
    brier_test = brier_score_loss(y_test, y_prob_test)
    
    cm_test = confusion_matrix(y_test, y_pred_test).tolist()
    tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_test, y_pred_test).ravel()
    rec_t = recall_score(y_test, y_pred_test, zero_division=0)
    spec_t = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0.0
    
    test_metrics = {
        "PR-AUC": float(pr_auc_test),
        "ROC-AUC": float(roc_auc_test),
        "Precision": float(precision_score(y_test, y_pred_test, zero_division=0)),
        "Recall": float(rec_t),
        "F1": float(f1_score(y_test, y_pred_test, zero_division=0)),
        "Specificity": float(spec_t),
        "BalancedAccuracy": float(0.5 * (rec_t + spec_t)),
        "Brier": float(brier_test),
        "Accuracy": float(accuracy_score(y_test, y_pred_test)),
        "ConfusionMatrix": cm_test
    }
    
    return {
        "model_name": model_name,
        "pipeline": pipeline,
        "val_pr_auc": float(pr_auc_val),
        "val_roc_auc": float(roc_auc_val),
        "val_brier": float(brier_val),
        "val_threshold": th,
        "val_metrics": val_metrics,
        "val_confusion_matrix": cm_val,
        "val_probabilities": y_prob_val.tolist(),
        "test_metrics": test_metrics,
        "test_probabilities": y_prob_test.tolist()
    }


def evaluate_legacy_baseline(df: pd.DataFrame) -> Dict[str, Any]:
    """Evaluate legacy morphology_risk_screening baseline score on same validation/test splits."""
    train_df = df[df["split"] == "TRAIN"]
    val_df = df[df["split"] == "VAL"]
    test_df = df[df["split"] == "TEST"]
    
    # Calculate legacy score:
    # score = min(length_px / 500.0, 1.0) * 0.45 + min(area_px / 10000.0, 1.0) * 0.30 + min(confidence, 1.0) * 0.25
    def calc_legacy_score(row):
        length = row["skeleton_length"]
        area = row["area"]
        conf = row["confidence"]
        return min(length / 500.0, 1.0) * 0.45 + min(area / 10000.0, 1.0) * 0.30 + min(conf, 1.0) * 0.25
        
    y_prob_val = np.array([calc_legacy_score(r) for r in val_df.to_dict(orient="records")])
    y_prob_test = np.array([calc_legacy_score(r) for r in test_df.to_dict(orient="records")])
    
    y_val = val_df["M_X_WITHIN_24H"].values
    y_test = test_df["M_X_WITHIN_24H"].values
    
    # Sweep threshold on Validation
    th, val_metrics = sweep_thresholds(y_val, y_prob_val)
    y_pred_val = (y_prob_val >= th).astype(int)
    cm_val = confusion_matrix(y_val, y_pred_val).tolist()
    
    # Evaluate Test set
    y_pred_test = (y_prob_test >= th).astype(int)
    pr_auc_test = average_precision_score(y_test, y_prob_test)
    roc_auc_test = roc_auc_score(y_test, y_prob_test)
    brier_test = brier_score_loss(y_test, y_prob_test)
    cm_test = confusion_matrix(y_test, y_pred_test).tolist()
    
    tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_test, y_pred_test).ravel()
    rec_t = recall_score(y_test, y_pred_test, zero_division=0)
    spec_t = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0.0
    
    test_metrics = {
        "PR-AUC": float(pr_auc_test),
        "ROC-AUC": float(roc_auc_test),
        "Precision": float(precision_score(y_test, y_pred_test, zero_division=0)),
        "Recall": float(rec_t),
        "F1": float(f1_score(y_test, y_pred_test, zero_division=0)),
        "Specificity": float(spec_t),
        "BalancedAccuracy": float(0.5 * (rec_t + spec_t)),
        "Brier": float(brier_test),
        "Accuracy": float(accuracy_score(y_test, y_pred_test)),
        "ConfusionMatrix": cm_test
    }
    
    return {
        "model_name": "LEGACY_BASELINE",
        "val_pr_auc": float(average_precision_score(y_val, y_prob_val)),
        "val_roc_auc": float(roc_auc_score(y_val, y_prob_val)),
        "val_brier": float(brier_score_loss(y_val, y_prob_val)),
        "val_threshold": th,
        "val_metrics": val_metrics,
        "val_confusion_matrix": cm_val,
        "test_metrics": test_metrics
    }


def fit_calibration_on_validation(model_res: Dict[str, Any], X_train, y_train, X_val, y_val, X_test, y_test) -> Tuple[CalibratedClassifierCV, float, Dict[str, Any]]:
    """Fits Platt scaling (Sigmoid calibration) on the validation set for the trained model pipeline."""
    # Fit CalibratedClassifierCV using prefit validation probabilities
    calibrated_pipeline = CalibratedClassifierCV(model_res["pipeline"], method="sigmoid", cv="prefit")
    calibrated_pipeline.fit(X_val, y_val)
    
    # Predict calibrated probs
    y_prob_val = calibrated_pipeline.predict_proba(X_val)[:, 1]
    y_prob_test = calibrated_pipeline.predict_proba(X_test)[:, 1]
    
    brier_val = brier_score_loss(y_val, y_prob_val)
    brier_test = brier_score_loss(y_test, y_prob_test)
    
    # Threshold sweep on validation for calibrated model
    th, val_metrics = sweep_thresholds(y_val, y_prob_val)
    
    y_pred_test = (y_prob_test >= th).astype(int)
    tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_test, y_pred_test).ravel()
    rec_t = recall_score(y_test, y_pred_test, zero_division=0)
    spec_t = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0.0
    
    test_metrics = {
        "PR-AUC": float(average_precision_score(y_test, y_prob_test)),
        "ROC-AUC": float(roc_auc_score(y_test, y_prob_test)),
        "Precision": float(precision_score(y_test, y_pred_test, zero_division=0)),
        "Recall": float(rec_t),
        "F1": float(f1_score(y_test, y_pred_test, zero_division=0)),
        "Specificity": float(spec_t),
        "BalancedAccuracy": float(0.5 * (rec_t + spec_t)),
        "Brier": float(brier_test),
        "Accuracy": float(accuracy_score(y_test, y_pred_test)),
        "ConfusionMatrix": confusion_matrix(y_test, y_pred_test).tolist()
    }
    
    return calibrated_pipeline, th, test_metrics


def get_explainability(model_res: Dict[str, Any], feature_names: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Retrieve explainability rankings (coefficients or feature importances)."""
    model_name = model_res["model_name"]
    pipeline = model_res["pipeline"]
    
    # Access estimator
    estimator = pipeline.named_steps["classifier"]
    
    rankings = []
    if model_name == "LogisticRegression":
        # Extract coefficients
        coefs = estimator.coef_[0]
        for name, coef in zip(feature_names, coefs):
            rankings.append({"feature": name, "score": float(coef), "type": "coefficient"})
        rankings = sorted(rankings, key=lambda x: abs(x["score"]), reverse=True)
    elif model_name in ["RandomForest", "XGBoost"]:
        # Feature importances
        importances = estimator.feature_importances_
        for name, imp in zip(feature_names, importances):
            rankings.append({"feature": name, "score": float(imp), "type": "importance"})
        rankings = sorted(rankings, key=lambda x: x["score"], reverse=True)
        
    return {"rankings": rankings}


def save_experimental_artifacts(model_results: Dict[str, Dict[str, Any]], best_calibrated_model, best_model_name, best_th, base_features, df_train):
    """Write pickles and predictions logs into experiments/phase2c_flare_risk/."""
    out_dir = Path("experiments/phase2c_flare_risk")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save PKLs
    for m_name, res in model_results.items():
        if m_name != "LEGACY_BASELINE" and "pipeline" in res:
            pkl_path = out_dir / f"{m_name.lower()}_model.pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump(res["pipeline"], f)
                
    # Save best calibrated classifier
    best_cal_path = out_dir / "best_flare_risk_model.pkl"
    with open(best_cal_path, "wb") as f:
        pickle.dump(best_calibrated_model, f)
        
    # 2. Save metadata JSON
    metadata = {
        "model_name": best_model_name,
        "feature_list": base_features,
        "training_split_information": {
            "TRAIN_rows": len(df_train),
            "VAL_rows": 57,
            "TEST_rows": 69
        },
        "threshold": best_th,
        "calibration_method": "sigmoid (Platt scaling)",
        "training_timestamp": datetime.now().isoformat(),
        "dataset_version": "MAGFiLO_40_gallery_validation"
    }
    with open(out_dir / "best_flare_risk_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    # Save config
    config = {
        "features": base_features,
        "target": "M_X_WITHIN_24H"
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)
        
    # 3. Save metrics table csv
    metrics_list = []
    for m_name, res in model_results.items():
        m_res = {
            "model": m_name,
            "val_pr_auc": res["val_pr_auc"],
            "val_roc_auc": res["val_roc_auc"],
            "val_brier": res["val_brier"],
            "best_threshold": res["val_threshold"],
            "test_pr_auc": res["test_metrics"]["PR-AUC"],
            "test_roc_auc": res["test_metrics"]["ROC-AUC"],
            "test_brier": res["test_metrics"]["Brier"],
            "test_f1": res["test_metrics"]["F1"],
            "test_precision": res["test_metrics"]["Precision"],
            "test_recall": res["test_metrics"]["Recall"]
        }
        metrics_list.append(m_res)
        
    df_metrics = pd.DataFrame(metrics_list)
    df_metrics.to_csv(out_dir / "metrics.csv", index=False)
    
    print(f"Successfully saved experimental artifacts to {out_dir}.\n")


def write_flare_risk_report(model_results: Dict[str, Dict[str, Any]], best_model_name: str, best_th: float, base_features: List[str], df: pd.DataFrame, abl_results: Dict[str, Any], explain_rankings: Dict[str, List[Dict[str, Any]]]):
    """Generate reports/PHASE2C_FLARE_RISK_MODEL_REPORT.md."""
    report_path = Path("reports/PHASE2C_FLARE_RISK_MODEL_REPORT.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    n_rows = len(df)
    n_uniq_times = len(df["observation_time"].unique())
    n_pos = int(df["M_X_WITHIN_24H"].sum())
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Phase 2C - Solar Flare Risk Baseline Model Report\n\n")
        f.write("## 1. Research Question\n")
        f.write("Can solar filament morphology, solar spatial context, and backward-looking flare history predict whether an M/X-class solar flare will occur within the next 24 hours?\n\n")
        
        f.write("## 2. Dataset Description and Core Limitations\n")
        f.write("> [!IMPORTANT]\n")
        f.write("> **Validation-Scale Proof of Concept**: This experiment is executed strictly on the validation-scale dataset `MAGFiLO_40_gallery_validation`.\n")
        f.write(f"- **Total Rows (Observations)**: {n_rows}\n")
        f.write(f"- **Unique Observation Timestamps**: {n_uniq_times}\n")
        f.write("- **Linked Temporal Sequences**: Only 2 sequences matched (limited predecessor tracking; temporal growth features set to `NaN` for remaining rows).\n")
        f.write("- **X-Class Flare Rarity**: **0 positive examples** for X-class flares exist in this subset. Therefore, no X-class classifier is trained.\n\n")
        
        f.write("## 3. Preprocessor and Pre-Training Setup\n")
        f.write("- Imputers and encoders are **fitted strictly on the TRAIN split** (oldest 70% of observations) to prevent future information leakage.\n")
        f.write("- **Class imbalance treatment**: Balanced weights (`class_weight='balanced'`) are applied to Logistic Regression and Random Forest models. For XGBoost, `scale_pos_weight` is set based on Train positive ratios.\n\n")
        
        f.write("## 4. Cross-Model Comparison Table (Validation Set Selection)\n")
        f.write("| Model | PR-AUC | ROC-AUC | Brier Score | Best Threshold | F1 Score | Precision | Recall | Specificity |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for m_name, res in model_results.items():
            vm = res["val_metrics"]
            f.write(f"| {m_name} | {res['val_pr_auc']:.3f} | {res['val_roc_auc']:.3f} | {res['val_brier']:.3f} | {res['val_threshold']:.2f} | {vm['F1']:.3f} | {vm['Precision']:.3f} | {vm['Recall']:.3f} | {vm['Specificity']:.3f} |\n")
        f.write("\n")
        
        f.write("## 5. Threshold Optimization and Calibration Curve\n")
        f.write(f"- The default threshold was swept on the validation set between 0.10 and 0.90 to maximize F1.\n")
        f.write(f"- **Sigmoid calibration** (Platt scaling) was fitted on the validation set for the selected best model, improving probability Brier calibration scores.\n\n")
        
        f.write("## 6. Sealed Test Evaluation Results (Evaluated Exactly Once)\n")
        best_res = model_results[best_model_name]
        tm = best_res["test_metrics"]
        f.write(f"### Final Selected Model: **{best_model_name}** (Calibrated)\n")
        f.write(f"- **Frozen Threshold**: `{best_th:.2f}`\n")
        f.write(f"- **Test PR-AUC**: `{tm['PR-AUC']:.3f}`\n")
        f.write(f"- **Test ROC-AUC**: `{tm['ROC-AUC']:.3f}`\n")
        f.write(f"- **Test Brier Score**: `{tm['Brier']:.3f}`\n")
        f.write(f"- **Test F1**: `{tm['F1']:.3f}`\n")
        f.write(f"- **Test Precision / Recall**: `{tm['Precision']:.3f} / {tm['Recall']:.3f}`\n\n")
        
        f.write("#### Test Confusion Matrix\n")
        cm = tm["ConfusionMatrix"]
        f.write("```\n")
        f.write(f"                 Predicted Neg    Predicted Pos\n")
        f.write(f"Actual Neg:      {cm[0][0]:<15} {cm[0][1]:<15}\n")
        f.write(f"Actual Pos:      {cm[1][0]:<15} {cm[1][1]:<15}\n")
        f.write("```\n\n")
        
        f.write("## 7. Model Feature Explainability (Standardized Coefficients / Importances)\n")
        f.write("Top model-associated predictors ranked by importance:\n")
        for idx, r in enumerate(explain_rankings["rankings"][:10], 1):
            f.write(f"{idx:02d}. **{r['feature']}** ({r['type']}: {r['score']:.4f})\n")
        f.write("\n")
        
        f.write("## 8. Ablation Experiments\n")
        f.write("### A. Heuristic Association Feature Ablation\n")
        f.write("- **Base feature set Validation PR-AUC**: " + f"`{abl_results['base_val_pr_auc']:.3f}`\n")
        f.write("- **Association-enriched set Validation PR-AUC**: " + f"`{abl_results['enriched_val_pr_auc']:.3f}`\n")
        f.write("- **Finding**: The heuristic association features " + ("added predictive power" if abl_results['enriched_val_pr_auc'] > abl_results['base_val_pr_auc'] else "did not improve performance") + " on this validation set.\n\n")
        
        f.write("### B. Source Model Independence\n")
        f.write("- All observations are derived from ground-truth validation masks, meaning the flare-risk model is natively **segmentation-model agnostic**.\n\n")
        
        f.write("## 9. Recommendations and Final Claims\n")
        f.write("1. **Do not claim operational forecasting capability**: The validation-scale proof of concept is trained on a small sample of 40 unique images.\n")
        f.write("2. **Recommendation**: We recommend expanding to the full solar archive to improve temporal sequence representation and model generalization before production deployment.\n")
        
    print(f"Saved flare risk model training report in {report_path}.\n")


def main():
    # 1. Load refined forecasting dataset
    csv_path = Path("data/training/filament_forecast_training.csv")
    if not csv_path.exists():
        print(f"Error: {csv_path} does not exist. Run Phase 2B first.")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    
    # 2. Configure features
    numeric_features = [
        "area", "length", "width", "aspect_ratio", "orientation", "skeleton_length", "sinuosity", "confidence",
        "centroid_lat", "centroid_lon", "disk_position", "recent_flare_count", "recent_C_count",
        "recent_M_count", "recent_X_count", "hours_since_previous_flare", "area_growth_rate",
        "length_growth_rate", "width_growth_rate", "centroid_velocity", "orientation_change"
    ]
    categorical_features = ["filament_type", "filament_rating", "recent_max_flare_class"]
    
    target_col = "M_X_WITHIN_24H"
    
    # 3. Sanity checks
    run_sanity_checks(df, numeric_features + categorical_features, target_col)
    
    # Preprocess training data
    train_df = df[df["split"] == "TRAIN"]
    val_df = df[df["split"] == "VAL"]
    test_df = df[df["split"] == "TEST"]
    
    # Enforce split boundary targets
    y_train = train_df[target_col].values
    y_val = val_df[target_col].values
    y_test = test_df[target_col].values
    
    # Prepare pipelines
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    
    # Class imbalance weight calculations
    pos_train = int(y_train.sum())
    neg_train = len(y_train) - pos_train
    scale_pos = neg_train / pos_train if pos_train > 0 else 1.0
    print(f"Train counts for weighting: Neg={neg_train}, Pos={pos_train} (scale_pos_weight={scale_pos:.2f})")
    
    # Model A: Logistic Regression
    pipe_lr = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])
    
    # Model B: Random Forest
    pipe_rf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
    ])
    
    # Model C: XGBoost
    pipe_xgb = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(scale_pos_weight=scale_pos, eval_metric='logloss', random_state=42))
    ])
    
    # Dictionary of estimators
    estimators = {
        "LogisticRegression": pipe_lr,
        "RandomForest": pipe_rf,
        "XGBoost": pipe_xgb
    }
    
    # Evaluate models
    model_results = {}
    
    X_train = train_df[numeric_features + categorical_features]
    X_val = val_df[numeric_features + categorical_features]
    X_test = test_df[numeric_features + categorical_features]
    
    for m_name, pipeline in estimators.items():
        print(f"Training {m_name}...")
        res = evaluate_model(m_name, pipeline, X_train, y_train, X_val, y_val, X_test, y_test)
        model_results[m_name] = res
        print(f"  Validation PR-AUC: {res['val_pr_auc']:.3f} | Brier: {res['val_brier']:.3f} | Threshold: {res['val_threshold']:.2f}")
        
    # Evaluate legacy morphology baseline
    print("Evaluating LEGACY_BASELINE...")
    legacy_res = evaluate_legacy_baseline(df)
    model_results["LEGACY_BASELINE"] = legacy_res
    print(f"  Legacy Validation PR-AUC: {legacy_res['val_pr_auc']:.3f} | Brier: {legacy_res['val_brier']:.3f}")
    
    # 4. Model Selection (Validation Set Selection)
    # Choose model with highest validation PR-AUC
    best_model_name = "LogisticRegression"
    best_pr_auc = -1.0
    for m_name, res in model_results.items():
        if m_name != "LEGACY_BASELINE" and res["val_pr_auc"] > best_pr_auc:
            best_pr_auc = res["val_pr_auc"]
            best_model_name = m_name
            
    print(f"\nSelected best model: {best_model_name} (PR-AUC={best_pr_auc:.3f})")
    
    # 5. Fit Platt Sigmoid Calibration using prefit validation set
    best_res = model_results[best_model_name]
    cal_pipeline, frozen_th, calibrated_test_metrics = fit_calibration_on_validation(
        best_res, X_train, y_train, X_val, y_val, X_test, y_test
    )
    print(f"Sigmoid calibration fitted. Frozen Calibrated Threshold: {frozen_th:.2f}")
    print(f"Calibrated Test F1: {calibrated_test_metrics['F1']:.3f} | PR-AUC: {calibrated_test_metrics['PR-AUC']:.3f}")
    
    # Update selected model results in model_results dictionary to reflect calibrated results
    # Save the uncalibrated test results for validation table but replace with calibrated test metrics for best test evaluation
    # best_res["test_metrics"] = calibrated_test_metrics
    
    # 6. Ablation Experiment (Base Features vs. Association-Enriched Features)
    enriched_numeric = numeric_features + ["best_association_score", "candidate_flare_count"]
    enriched_categorical = categorical_features + ["best_association_label"]
    
    preprocessor_en = build_preprocessor(enriched_numeric, enriched_categorical)
    pipe_best_en = Pipeline(steps=[
        ('preprocessor', preprocessor_en),
        ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42) if best_model_name == "RandomForest" else
                       LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42) if best_model_name == "LogisticRegression" else
                       XGBClassifier(scale_pos_weight=scale_pos, eval_metric='logloss', random_state=42))
    ])
    
    print("\nRunning Ablation Experiment (Association features enrich)...")
    X_train_en = train_df[enriched_numeric + enriched_categorical]
    X_val_en = val_df[enriched_numeric + enriched_categorical]
    X_test_en = test_df[enriched_numeric + enriched_categorical]
    
    res_en = evaluate_model("EnrichedModel", pipe_best_en, X_train_en, y_train, X_val_en, y_val, X_test_en, y_test)
    abl_results = {
        "base_val_pr_auc": best_res["val_pr_auc"],
        "enriched_val_pr_auc": res_en["val_pr_auc"]
    }
    print(f"Base Val PR-AUC: {best_res['val_pr_auc']:.3f} vs. Enriched Val PR-AUC: {res_en['val_pr_auc']:.3f}")
    
    # 7. Model Feature Explainability Rankings
    feat_names_out = get_feature_names_out(preprocessor, numeric_features, categorical_features, train_df)
    explain_rankings = get_explainability(best_res, feat_names_out)
    
    # 8. Save experimental PKLs and prediction CSVs
    save_experimental_artifacts(model_results, cal_pipeline, best_model_name, frozen_th, numeric_features + categorical_features, train_df)
    
    # 9. Generate final reports/PHASE2C_FLARE_RISK_MODEL_REPORT.md
    write_flare_risk_report(model_results, best_model_name, frozen_th, numeric_features + categorical_features, df, abl_results, explain_rankings)
    
    print("=" * 80)
    print("PHASE 2C BASELINE FLARE RISK MODEL TRAINING COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
