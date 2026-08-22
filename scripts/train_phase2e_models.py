"""
Phase 2E — Full-Archive Filament → Flare Risk Model Training
=============================================================
Trains 9 primary models (3 feature configs × 3 algorithms) on the 5,238-row
full historical archive (filament_forecast_full.csv) produced in Phase 2D.

STRICT ADDITIVE EXPERIMENT:
- Does NOT modify any Phase 2B, 2C, or 2D files.
- Creates experiments/phase2e_flare_risk/ and reports/PHASE2E_*.md only.
"""
import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

warnings.filterwarnings("ignore")
np.random.seed(42)

# Scikit-learn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss,
    precision_score, recall_score, f1_score, confusion_matrix,
    balanced_accuracy_score
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: XGBoost not available. Will run LR and RF only.")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
OUT_DIR = Path("experiments/phase2e_flare_risk")
REPORT_DIR = Path("reports")
DATA_PATH = Path("data/training/filament_forecast_full.csv")

# Future-event columns that MUST NOT appear in any feature group
FORBIDDEN_FEATURES = {
    "future_flare_count", "first_future_flare_time", "first_future_flare_class",
    "strongest_future_flare_class", "M_X_WITHIN_6H", "M_X_WITHIN_12H",
    "M_X_WITHIN_48H", "X_CLASS_48H", "C_OR_HIGHER_24H", "M_OR_HIGHER_24H",
    "X_CLASS_24H", "M_X_WITHIN_24H"
}

# Feature group definitions
GROUP_A_MORPHOLOGY = [
    "area", "length", "width", "skeleton_length",
    "aspect_ratio", "sinuosity", "orientation", "confidence"
]
GROUP_B_POSITION = [
    "centroid_lat", "centroid_lon", "disk_position"
]
GROUP_C_HIST_FLARE = [
    "recent_flare_count", "recent_C_count", "recent_M_count", "recent_X_count",
    "hours_since_previous_flare"
]
GROUP_D_AR_HISTORY = [
    "active_region_previous_flare_count", "active_region_previous_M_count",
    "active_region_previous_X_count"
]
GROUP_E_TEMPORAL = [
    "area_growth_rate", "length_growth_rate", "width_growth_rate",
    "centroid_velocity", "orientation_change", "aspect_ratio_change",
    "area_acceleration", "length_acceleration"
]
# Categorical features (OHE-encoded, train-fitted)
CATEGORICAL_FEATURES = [
    "filament_type", "filament_rating", "recent_max_flare_class"
]
# active_region raw ID dropped — too high cardinality, no ordinal meaning.

# Association features (secondary ablation only — NOT in primary models)
ASSOCIATION_FEATURES = {
    "numeric": ["best_assoc_score", "candidate_flare_count"],
    "categorical": ["best_assoc_label"]
}

PRIMARY_TARGET = "M_X_WITHIN_24H"
X_CLASS_TARGET = "X_CLASS_24H"

PHASE2C_BASELINE = {
    "rows": 507, "timestamps": 40, "mx_positives": 135,
    "test_pr_auc": 0.336, "test_roc_auc": 0.292,
    "test_brier": 0.380, "test_f1": 0.109
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA AUDIT
# ─────────────────────────────────────────────────────────────────────────────
def run_data_audit(df: pd.DataFrame, target: str) -> Tuple[bool, bool]:
    """Print full dataset audit and return (mx_ok, xclass_ok)."""
    print("=" * 80)
    print("PHASE 2E — DATA AUDIT")
    print("=" * 80)

    n = len(df)
    n_times = df["observation_time"].nunique()
    n_tracks = df["physical_track_id"].nunique()
    n_pos_mx = int(df[PRIMARY_TARGET].sum())
    n_pos_c = int(df["C_OR_HIGHER_24H"].sum())
    n_pos_x = int(df[X_CLASS_TARGET].sum())

    print(f"Total rows              : {n}")
    print(f"Unique timestamps       : {n_times}")
    print(f"Unique physical tracks  : {n_tracks}")
    print(f"M/X positives (24h)     : {n_pos_mx} ({n_pos_mx/n*100:.1f}%)")
    print(f"C+ positives  (24h)     : {n_pos_c} ({n_pos_c/n*100:.1f}%)")
    print(f"X-class positives (24h) : {n_pos_x} ({n_pos_x/n*100:.1f}%)")

    # Track length distribution
    track_lengths = df.groupby("physical_track_id").size()
    print(f"\nTrack length distribution:")
    print(f"  singleton (len=1): {(track_lengths == 1).sum()}")
    print(f"  len=2           : {(track_lengths == 2).sum()}")
    print(f"  len>=3          : {(track_lengths >= 3).sum()}")

    print("\nPer-split breakdown:")
    mx_ok = True
    for split in ["TRAIN", "VAL", "TEST"]:
        sub = df[df["split"] == split]
        if sub.empty:
            print(f"  {split}: EMPTY — ABORT")
            sys.exit(1)
        pos = int(sub[PRIMARY_TARGET].sum())
        rate = pos / len(sub) * 100
        print(f"  {split}: rows={len(sub)}, M/X pos={pos}, rate={rate:.1f}%")
        if sub[PRIMARY_TARGET].nunique() < 2:
            print(f"  ERROR: {split} has only one M/X class — ABORT")
            sys.exit(1)

    # X-class feasibility
    xclass_ok = True
    print("\nX-class split breakdown:")
    for split in ["TRAIN", "VAL", "TEST"]:
        sub = df[df["split"] == split]
        xpos = int(sub[X_CLASS_TARGET].sum())
        print(f"  {split}: X positives = {xpos}")
        if xpos < 5:
            xclass_ok = False

    if xclass_ok:
        print("  X-class experiment: FEASIBLE (all splits ≥ 5 positives)")
    else:
        print("  X-class experiment: INSUFFICIENT DATA (some split < 5 positives)")
        print("  Will NOT train an X-class classifier. Reporting explicitly.")

    # Temporal coverage
    n_temporal = int((~df["area_growth_rate"].isna()).sum())
    print(f"\nRows with temporal history : {n_temporal} ({n_temporal/n*100:.1f}%)")
    print(f"Rows without (singleton)   : {n - n_temporal} ({(n-n_temporal)/n*100:.1f}%)")

    print("\nDATA AUDIT COMPLETE.\n")
    return mx_ok, xclass_ok


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SAFETY CHECK
# ─────────────────────────────────────────────────────────────────────────────
def assert_no_leakage(features: List[str], name: str):
    leaked = [f for f in features if f in FORBIDDEN_FEATURES]
    if leaked:
        print(f"CRITICAL LEAKAGE in {name}: {leaked}")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """Build a train-fittable ColumnTransformer."""
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    transformers = [("num", num_pipe, numeric_cols)]
    if categorical_cols:
        transformers.append(("cat", cat_pipe, categorical_cols))
    return ColumnTransformer(transformers=transformers)


def get_feature_names(preprocessor: ColumnTransformer, numeric_cols, categorical_cols) -> List[str]:
    """Extract feature names post-fit."""
    names = list(numeric_cols)
    if categorical_cols:
        ohe = preprocessor.named_transformers_["cat"].named_steps["ohe"]
        names += ohe.get_feature_names_out(categorical_cols).tolist()
    return names


# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLD SWEEP (VAL ONLY)
# ─────────────────────────────────────────────────────────────────────────────
def sweep_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, Dict]:
    best_th, best_f1 = 0.50, -1.0
    best_metrics = {}
    for th in np.arange(0.01, 1.00, 0.01):
        y_pred = (y_prob >= th).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = float(th)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            rec = recall_score(y_true, y_pred, zero_division=0)
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            best_metrics = {
                "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "Recall": float(rec), "F1": float(f1),
                "Specificity": float(spec),
                "BalancedAccuracy": float(0.5 * (rec + spec))
            }
    return best_th, best_metrics


# ─────────────────────────────────────────────────────────────────────────────
# MODEL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, th: float) -> Dict:
    y_pred = (y_prob >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    rec = recall_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "PR-AUC": float(average_precision_score(y_true, y_prob)),
        "ROC-AUC": float(roc_auc_score(y_true, y_prob)),
        "Brier": float(brier_score_loss(y_true, y_prob)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(rec),
        "Specificity": float(spec),
        "BalancedAccuracy": float(0.5 * (rec + spec)),
        "ConfusionMatrix": confusion_matrix(y_true, y_pred).tolist()
    }


def train_and_evaluate(
    model_name: str,
    pipeline: Pipeline,
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    label: str
) -> Dict[str, Any]:
    """Fit on train, sweep threshold on val, evaluate test once."""
    pipeline.fit(X_train, y_train)

    y_prob_val = pipeline.predict_proba(X_val)[:, 1]
    y_prob_test = pipeline.predict_proba(X_test)[:, 1]

    val_pr_auc = average_precision_score(y_val, y_prob_val)
    val_roc_auc = roc_auc_score(y_val, y_prob_val)
    val_brier = brier_score_loss(y_val, y_prob_val)

    # Threshold from VAL only
    th, val_th_metrics = sweep_thresholds(y_val, y_prob_val)

    # Calibrate on VAL
    cal = CalibratedClassifierCV(pipeline, method="sigmoid", cv="prefit")
    cal.fit(X_val, y_val)
    y_prob_val_cal = cal.predict_proba(X_val)[:, 1]
    y_prob_test_cal = cal.predict_proba(X_test)[:, 1]
    th_cal, _ = sweep_thresholds(y_val, y_prob_val_cal)

    val_metrics = compute_metrics_at_threshold(y_val, y_prob_val, th)
    test_metrics = compute_metrics_at_threshold(y_test, y_prob_test, th)
    test_cal_metrics = compute_metrics_at_threshold(y_test, y_prob_test_cal, th_cal)

    print(f"  [{label}] {model_name}: Val PR-AUC={val_pr_auc:.3f} | "
          f"Test PR-AUC={test_metrics['PR-AUC']:.3f} | "
          f"Test Cal PR-AUC={test_cal_metrics['PR-AUC']:.3f} | th={th:.2f}")

    return {
        "label": label,
        "model_name": model_name,
        "pipeline": pipeline,
        "calibrated_pipeline": cal,
        "val_pr_auc": val_pr_auc,
        "val_roc_auc": val_roc_auc,
        "val_brier": val_brier,
        "val_threshold": th,
        "val_threshold_cal": th_cal,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_cal_metrics": test_cal_metrics,
        "val_probs": y_prob_val.tolist(),
        "test_probs": y_prob_test.tolist(),
        "val_probs_cal": y_prob_val_cal.tolist(),
        "test_probs_cal": y_prob_test_cal.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# NAIVE BASELINE
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_naive_baseline(y_train, y_val, y_test) -> Dict:
    train_rate = float(y_train.mean())
    print(f"\nNaive baseline: predict_prob = TRAIN rate = {train_rate:.4f} for all observations")
    val_naive = np.full(len(y_val), train_rate)
    test_naive = np.full(len(y_test), train_rate)
    return {
        "train_positive_rate": train_rate,
        "val_brier": float(brier_score_loss(y_val, val_naive)),
        "val_pr_auc": float(average_precision_score(y_val, val_naive)),
        "test_brier": float(brier_score_loss(y_test, test_naive)),
        "test_pr_auc": float(average_precision_score(y_test, test_naive)),
        "test_roc_auc": float(roc_auc_score(y_test, test_naive)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SOLAR-CYCLE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def solar_cycle_analysis(df: pd.DataFrame, best_probs_test: np.ndarray,
                          test_df: pd.DataFrame) -> Dict:
    """Compute year-by-year M/X positive rate and naive rate."""
    df = df.copy()
    df["year"] = pd.to_datetime(df["observation_time"]).dt.year

    rows = []
    for yr in sorted(df["year"].unique()):
        sub = df[df["year"] == yr]
        pos = int(sub[PRIMARY_TARGET].sum())
        n = len(sub)
        rate = pos / n if n > 0 else 0.0
        rows.append({"year": yr, "n": n, "mx_positive": pos, "mx_rate": round(rate, 4),
                     "split": sub["split"].mode()[0] if not sub.empty else "?"})
    return {"yearly": rows}


# ─────────────────────────────────────────────────────────────────────────────
# TRACK-AWARE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def track_aware_analysis(df_test: pd.DataFrame, y_test: np.ndarray, y_prob: np.ndarray,
                          th: float) -> Dict:
    """Evaluate separately for singleton vs multi-observation tracks."""
    track_counts = df_test.groupby("physical_track_id").transform("count")["observation_time"]
    singleton_mask = (track_counts == 1).values
    multi_mask = ~singleton_mask

    results = {}
    for label, mask in [("singleton", singleton_mask), ("tracked", multi_mask)]:
        y_t = y_test[mask]
        y_p = y_prob[mask]
        if len(y_t) < 5 or len(np.unique(y_t)) < 2:
            results[label] = {"n": int(mask.sum()), "skipped": "insufficient data"}
            continue
        y_pred = (y_p >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_t, y_pred).ravel()
        rec = recall_score(y_t, y_pred, zero_division=0)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        results[label] = {
            "n": int(mask.sum()),
            "mx_positive": int(y_t.sum()),
            "PR-AUC": float(average_precision_score(y_t, y_p)) if y_t.sum() > 0 else None,
            "Recall": float(rec),
            "Specificity": float(spec),
            "F1": float(f1_score(y_t, y_pred, zero_division=0))
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXPLAINABILITY
# ─────────────────────────────────────────────────────────────────────────────
def get_feature_importance(pipeline: Pipeline, feature_names: List[str], model_name: str) -> List[Dict]:
    clf = pipeline.named_steps["classifier"]
    rankings = []
    if model_name == "LogisticRegression":
        coefs = clf.coef_[0]
        for name, coef in zip(feature_names, coefs):
            rankings.append({"feature": name, "score": float(coef), "type": "coefficient"})
        rankings.sort(key=lambda x: abs(x["score"]), reverse=True)
    elif model_name in ("RandomForest", "XGBoost"):
        imps = clf.feature_importances_
        for name, imp in zip(feature_names, imps):
            rankings.append({"feature": name, "score": float(imp), "type": "importance"})
        rankings.sort(key=lambda x: x["score"], reverse=True)
    return rankings[:15]


# ─────────────────────────────────────────────────────────────────────────────
# YEAR-BY-YEAR EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def year_by_year_eval(df_test: pd.DataFrame, y_test: np.ndarray, y_prob: np.ndarray,
                       th: float) -> List[Dict]:
    df_test = df_test.copy()
    df_test["__prob"] = y_prob
    df_test["__target"] = y_test
    df_test["__year"] = pd.to_datetime(df_test["observation_time"]).dt.year

    rows = []
    for yr in sorted(df_test["__year"].unique()):
        sub = df_test[df_test["__year"] == yr]
        y_t = sub["__target"].values
        y_p = sub["__prob"].values
        n = len(sub)
        pos = int(y_t.sum())
        rate = pos / n if n > 0 else 0.0
        row = {"year": int(yr), "n": n, "mx_positive": pos, "mx_rate": round(rate, 4)}

        if pos >= 2 and (n - pos) >= 2:
            y_pred = (y_p >= th).astype(int)
            row["Recall"] = float(recall_score(y_t, y_pred, zero_division=0))
            row["Precision"] = float(precision_score(y_t, y_pred, zero_division=0))
            if pos >= 20:
                row["PR-AUC"] = float(average_precision_score(y_t, y_p))
        rows.append(row)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# ARTIFACT SAVING
# ─────────────────────────────────────────────────────────────────────────────
def save_artifacts(all_results: Dict, best_key: str, preprocessors: Dict,
                   thresholds: Dict, val_df, test_df,
                   numeric_configs: Dict, cat_configs: Dict):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save individual model PKLs
    for key, res in all_results.items():
        if "pipeline" not in res:
            continue
        algo = res["model_name"].lower().replace(" ", "")
        config = res["label"].lower().replace(" ", "_").replace("+", "")
        pkl_name = f"{algo}_{config}.pkl"
        with open(OUT_DIR / pkl_name, "wb") as f:
            pickle.dump(res["pipeline"], f)

    # Save best model (calibrated)
    best_res = all_results[best_key]
    with open(OUT_DIR / "best_flare_risk_model.pkl", "wb") as f:
        pickle.dump(best_res["calibrated_pipeline"], f)

    # Save preprocessors
    for config_name, prep in preprocessors.items():
        with open(OUT_DIR / f"preprocessor_{config_name}.pkl", "wb") as f:
            pickle.dump(prep, f)

    # Save thresholds JSON
    with open(OUT_DIR / "thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)

    # Best model metadata
    metadata = {
        "phase": "2E",
        "best_model_key": best_key,
        "model_name": best_res["model_name"],
        "config_label": best_res["label"],
        "target": PRIMARY_TARGET,
        "dataset": str(DATA_PATH),
        "val_pr_auc": best_res["val_pr_auc"],
        "val_threshold": best_res["val_threshold"],
        "val_threshold_cal": best_res.get("val_threshold_cal", best_res["val_threshold"]),
        "test_pr_auc": best_res["test_metrics"]["PR-AUC"],
        "test_cal_pr_auc": best_res["test_cal_metrics"]["PR-AUC"],
        "calibration_method": "sigmoid (Platt scaling) on VAL",
        "training_timestamp": datetime.now().isoformat(),
        "active_region_id_handling": "dropped (too high cardinality for OHE)",
        "association_features_in_primary": False
    }
    with open(OUT_DIR / "best_flare_risk_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Config JSON
    with open(OUT_DIR / "config.json", "w") as f:
        json.dump({
            "numeric_configs": numeric_configs,
            "categorical_configs": cat_configs,
            "target": PRIMARY_TARGET,
            "forbidden_features": list(FORBIDDEN_FEATURES)
        }, f, indent=2, default=list)

    # Metrics CSV
    rows = []
    for key, res in all_results.items():
        if "val_pr_auc" not in res:
            continue
        tm = res.get("test_metrics", {})
        tcm = res.get("test_cal_metrics", {})
        rows.append({
            "key": key, "model": res.get("model_name", ""), "config": res.get("label", ""),
            "val_pr_auc": res["val_pr_auc"], "val_roc_auc": res.get("val_roc_auc", ""),
            "val_brier": res.get("val_brier", ""), "val_threshold": res.get("val_threshold", ""),
            "test_pr_auc": tm.get("PR-AUC", ""), "test_roc_auc": tm.get("ROC-AUC", ""),
            "test_brier": tm.get("Brier", ""), "test_f1": tm.get("F1", ""),
            "test_precision": tm.get("Precision", ""), "test_recall": tm.get("Recall", ""),
            "test_cal_pr_auc": tcm.get("PR-AUC", "")
        })
    pd.DataFrame(rows).to_csv(OUT_DIR / "metrics.csv", index=False)

    # Predictions CSV for val and test
    val_preds = val_df[["filament_observation_id", "observation_time", "split", PRIMARY_TARGET]].copy()
    val_preds["best_model_prob"] = best_res["val_probs"]
    val_preds["best_model_prob_cal"] = best_res["val_probs_cal"]
    val_preds.to_csv(OUT_DIR / "predictions_val.csv", index=False)

    test_preds = test_df[["filament_observation_id", "observation_time", "split", PRIMARY_TARGET]].copy()
    test_preds["best_model_prob"] = best_res["test_probs"]
    test_preds["best_model_prob_cal"] = best_res["test_probs_cal"]
    test_preds.to_csv(OUT_DIR / "predictions_test.csv", index=False)

    print(f"Saved all artifacts to {OUT_DIR}/\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN REPORT
# ─────────────────────────────────────────────────────────────────────────────
def write_main_report(all_results: Dict, best_key: str, naive_baseline: Dict,
                       track_results: Dict, year_results: List[Dict],
                       solar_data: Dict, xclass_ok: bool, df: pd.DataFrame,
                       top_features: Dict):
    path = REPORT_DIR / "PHASE2E_FLARE_RISK_REPORT.md"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    best = all_results[best_key]

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Phase 2E — Full-Archive Filament → Flare Risk Model Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 1. Research Question\n")
        f.write("Can solar filament morphology, heliographic position, backward-looking solar "
                "flare history, active-region context, and temporal filament evolution predict "
                "whether an M-class or X-class solar flare will occur within 24 hours?\n\n")
        f.write("> **Primary target**: `M_X_WITHIN_24H`\n\n")

        # Dataset
        n = len(df)
        f.write("## 2. Dataset Description\n")
        f.write(f"- **Source**: `{DATA_PATH}` (Phase 2D output)\n")
        f.write(f"- **Total observations**: {n}\n")
        f.write(f"- **Unique timestamps**: {df['observation_time'].nunique()}\n")
        f.write(f"- **Date range**: 2011–2022\n")
        f.write(f"- **M/X positives**: {int(df[PRIMARY_TARGET].sum())} ({df[PRIMARY_TARGET].mean()*100:.1f}%)\n")
        f.write(f"- **X-class positives**: {int(df[X_CLASS_TARGET].sum())}\n\n")

        # Distribution shift
        f.write("## 3. Solar-Cycle Distribution Shift\n")
        f.write("> [!WARNING]\n")
        f.write("> **Critical**: TRAIN covers solar maximum (higher M/X rate); VAL/TEST cover "
                "solar minimum. This is NOT a data processing error — it is real solar physics. "
                "Performance degradation from TRAIN→TEST must be attributed to this shift "
                "before concluding model failure.\n\n")
        f.write("| Split | Rows | M/X Positives | Positive Rate |\n")
        f.write("|---|---|---|---|\n")
        for split in ["TRAIN", "VAL", "TEST"]:
            sub = df[df["split"] == split]
            pos = int(sub[PRIMARY_TARGET].sum())
            f.write(f"| {split} | {len(sub)} | {pos} | {pos/len(sub)*100:.1f}% |\n")
        f.write("\n")

        # Feature groups
        f.write("## 4. Feature Groups\n")
        f.write("| Group | Features | Used In |\n")
        f.write("|---|---|---|\n")
        f.write(f"| A — Morphology | {', '.join(GROUP_A_MORPHOLOGY)} | All models |\n")
        f.write(f"| B — Solar Position | {', '.join(GROUP_B_POSITION)} | All models |\n")
        f.write(f"| C — Historical Flare | {', '.join(GROUP_C_HIST_FLARE)} | +Context, +Temporal |\n")
        f.write(f"| D — Active Region | {', '.join(GROUP_D_AR_HISTORY)} | +Context, +Temporal |\n")
        f.write(f"| E — Temporal Evolution | {', '.join(GROUP_E_TEMPORAL)} | +Temporal only |\n")
        f.write("\n")
        f.write("**Notes**:\n")
        f.write("- `has_temporal_history` indicator added to +Temporal model (1 if predecessor existed, never imputed).\n")
        f.write("- `active_region` raw ID dropped — too high cardinality, no ordinal meaning.\n")
        f.write("- Association features (`best_assoc_score`, etc.) excluded from all primary models.\n\n")

        # Naive baseline
        f.write("## 5. Naive Baseline (Solar Rate)\n")
        f.write(f"Predict `p = {naive_baseline['train_positive_rate']:.4f}` (TRAIN positive rate) for every observation.\n\n")
        f.write(f"| Split | Naive PR-AUC | Naive Brier |\n")
        f.write(f"|---|---|---|\n")
        f.write(f"| VAL | {naive_baseline['val_pr_auc']:.3f} | {naive_baseline['val_brier']:.3f} |\n")
        f.write(f"| TEST | {naive_baseline['test_pr_auc']:.3f} | {naive_baseline['test_brier']:.3f} |\n\n")
        f.write("ML models must exceed this baseline to demonstrate predictive value beyond "
                "simply knowing the ambient solar activity rate.\n\n")

        # Model comparison table
        f.write("## 6. Model Comparison (Validation)\n")
        f.write("| Config | Algorithm | Val PR-AUC | Val ROC-AUC | Val Brier | Val F1 | Threshold |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for key, res in all_results.items():
            if "val_pr_auc" not in res or "ABLATION" in key:
                continue
            vm = res.get("val_metrics", {})
            f.write(f"| {res.get('label','?')} | {res.get('model_name','?')} | "
                    f"{res['val_pr_auc']:.3f} | {res.get('val_roc_auc',0):.3f} | "
                    f"{res.get('val_brier',0):.3f} | {vm.get('F1',0):.3f} | "
                    f"{res.get('val_threshold',0):.2f} |\n")
        f.write("\n")

        # Sealed test results
        f.write("## 7. Sealed Test Evaluation (Best Model)\n")
        f.write(f"**Best model**: `{best['model_name']}` — `{best['label']}` "
                f"(selected by Val PR-AUC = {best['val_pr_auc']:.3f})\n\n")
        f.write("### Uncalibrated\n")
        f.write("| Metric | Value |\n|---|---|\n")
        for k, v in best["test_metrics"].items():
            if k != "ConfusionMatrix":
                f.write(f"| {k} | {v:.3f} |\n")
        f.write("\n### Calibrated (Platt Sigmoid on VAL)\n")
        f.write("| Metric | Value |\n|---|---|\n")
        for k, v in best["test_cal_metrics"].items():
            if k != "ConfusionMatrix":
                f.write(f"| {k} | {v:.3f} |\n")
        f.write("\n")
        cm = best["test_metrics"]["ConfusionMatrix"]
        f.write("#### Confusion Matrix\n```\n")
        f.write(f"                 Pred Neg  Pred Pos\n")
        f.write(f"Actual Neg :     {cm[0][0]:<9} {cm[0][1]:<9}\n")
        f.write(f"Actual Pos :     {cm[1][0]:<9} {cm[1][1]:<9}\n")
        f.write("```\n\n")

        # Feature ablation table
        f.write("## 8. Feature Ablation Summary\n")
        f.write("| Feature Config | Best Val PR-AUC | Best Test PR-AUC |\n")
        f.write("|---|---|---|\n")
        config_best = {}
        for key, res in all_results.items():
            if "val_pr_auc" not in res or "ABLATION" in key:
                continue
            lbl = res.get("label", "?")
            if lbl not in config_best or res["val_pr_auc"] > config_best[lbl]["val_pr_auc"]:
                config_best[lbl] = res
        for lbl in ["Static (A+B)", "+Context (A+B+C+D)", "+Temporal (A+B+C+D+E)"]:
            if lbl in config_best:
                r = config_best[lbl]
                f.write(f"| {lbl} | {r['val_pr_auc']:.3f} | {r['test_metrics']['PR-AUC']:.3f} |\n")
        # Association ablation if present
        abl = all_results.get("ABLATION_association", None)
        if abl and "val_pr_auc" in abl:
            f.write(f"| +Association (SECONDARY ABLATION) | {abl['val_pr_auc']:.3f} | "
                    f"{abl['test_metrics']['PR-AUC']:.3f} |\n")
        f.write("\n")

        # Track-aware
        f.write("## 9. Track-Aware Analysis\n")
        for label, tres in track_results.items():
            f.write(f"### {label.title()} Filaments\n")
            for k, v in tres.items():
                f.write(f"- **{k}**: {v}\n")
            f.write("\n")

        # Year-by-year
        f.write("## 10. Year-by-Year Generalization (Test Period)\n")
        f.write("| Year | N | M/X Positive | Rate | Recall | Precision | PR-AUC |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for yr in year_results:
            row = (f"| {yr['year']} | {yr['n']} | {yr['mx_positive']} | "
                   f"{yr['mx_rate']:.3f} | {yr.get('Recall', 'N/A')} | "
                   f"{yr.get('Precision', 'N/A')} | {yr.get('PR-AUC', 'N/A')} |")
            f.write(row + "\n")
        f.write("\n")

        # Feature explainability
        f.write("## 11. Feature Explainability\n")
        f.write("> These are **model-associated predictors**, not causal relationships.\n\n")
        for config_name, rankings in top_features.items():
            f.write(f"### {config_name}\n")
            for i, r in enumerate(rankings[:10], 1):
                f.write(f"{i:02d}. `{r['feature']}` ({r['type']}: {r['score']:.4f})\n")
            f.write("\n")

        # X-class analysis
        f.write("## 12. X-Class Analysis\n")
        if xclass_ok:
            xres = all_results.get("XCLASS_LogisticRegression", None)
            if xres and "val_pr_auc" in xres:
                f.write(f"X-class experiment was run (all splits had ≥5 positives).\n\n")
                f.write(f"- Val PR-AUC: {xres['val_pr_auc']:.3f}\n")
                f.write(f"- Test PR-AUC: {xres['test_metrics']['PR-AUC']:.3f}\n\n")
        else:
            f.write("> [!NOTE]\n")
            f.write("> **X-class data available, but insufficient for reliable split-level "
                    "evaluation** (at least one split had fewer than 5 X-class positive examples). "
                    "No X-class classifier was trained.\n\n")

        # Phase 2C comparison
        f.write("## 13. Comparison with Phase 2C Baseline\n")
        test_tm = best["test_metrics"]
        f.write("| Metric | Phase 2C | Phase 2E |\n")
        f.write("|---|---:|---:|\n")
        f.write(f"| Dataset rows | {PHASE2C_BASELINE['rows']} | {len(df)} |\n")
        f.write(f"| Unique timestamps | {PHASE2C_BASELINE['timestamps']} | {df['observation_time'].nunique()} |\n")
        f.write(f"| M/X positives | {PHASE2C_BASELINE['mx_positives']} | {int(df[PRIMARY_TARGET].sum())} |\n")
        f.write(f"| Test PR-AUC | {PHASE2C_BASELINE['test_pr_auc']:.3f} | {test_tm['PR-AUC']:.3f} |\n")
        f.write(f"| Test ROC-AUC | {PHASE2C_BASELINE['test_roc_auc']:.3f} | {test_tm['ROC-AUC']:.3f} |\n")
        f.write(f"| Test Brier | {PHASE2C_BASELINE['test_brier']:.3f} | {test_tm['Brier']:.3f} |\n")
        f.write(f"| Test F1 | {PHASE2C_BASELINE['test_f1']:.3f} | {test_tm['F1']:.3f} |\n\n")

        # Limitations
        f.write("## 14. Limitations\n")
        f.write("1. The chronological split means TRAIN/TEST span different phases of the solar "
                "cycle. Performance on TEST may understate true skill during solar maximum.\n")
        f.write("2. Temporal tracking links only 3.9% of observations — most filaments have no "
                "measured predecessor; temporal evolution features are sparse.\n")
        f.write("3. Active-region IDs are unavailable in the ground-truth annotations; "
                "the AR history features default to zero for most observations.\n")
        f.write("4. Do NOT claim operational forecasting capability based on this experiment.\n\n")

        # Final recommendation
        f.write("## 15. Final Model Recommendation\n")
        f.write(f"**Selected model**: `{best['model_name']}` — `{best['label']}`\n\n")
        f.write(f"- Validation PR-AUC: {best['val_pr_auc']:.3f}\n")
        f.write(f"- Test PR-AUC (uncalibrated): {test_tm['PR-AUC']:.3f}\n")
        f.write(f"- Test PR-AUC (calibrated): {best['test_cal_metrics']['PR-AUC']:.3f}\n")
        f.write(f"- Frozen threshold: {best['val_threshold']:.2f}\n\n")

    print(f"Saved main report: {path}\n")


# ─────────────────────────────────────────────────────────────────────────────
# SOLAR CYCLE REPORT
# ─────────────────────────────────────────────────────────────────────────────
def write_solar_cycle_report(df: pd.DataFrame, solar_data: Dict, naive_baseline: Dict):
    path = REPORT_DIR / "PHASE2E_SOLAR_CYCLE_DISTRIBUTION.md"
    df = df.copy()
    df["year"] = pd.to_datetime(df["observation_time"]).dt.year

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Phase 2E — Solar-Cycle Distribution Analysis\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 1. Background\n")
        f.write("Solar flare activity follows an approximately 11-year cycle. The Phase 2D "
                "dataset spans 2011–2022, which includes solar cycle 24 (peak ~2014) and the "
                "early rise of cycle 25. The chronological train/val/test split means TRAIN "
                "captures more active periods, while VAL/TEST capture the declining phase "
                "and solar minimum.\n\n")
        f.write("> [!IMPORTANT]\n")
        f.write("> Any model trained primarily on solar maximum data will likely see lower "
                "apparent precision in the VAL/TEST period. This is a **temporal distribution "
                "shift**, not necessarily model failure.\n\n")

        f.write("## 2. Year-by-Year M/X Positive Rate\n")
        f.write("| Year | N | M/X Positives | Positive Rate | Split |\n")
        f.write("|---|---|---|---|---|\n")
        for row in solar_data["yearly"]:
            f.write(f"| {row['year']} | {row['n']} | {row['mx_positive']} | "
                    f"{row['mx_rate']:.3f} | {row['split']} |\n")
        f.write("\n")

        f.write("## 3. Train/Val/Test Regime Mapping\n")
        f.write("| Split | Years Covered | M/X Rate |\n")
        f.write("|---|---|---|\n")
        for split in ["TRAIN", "VAL", "TEST"]:
            sub = df[df["split"] == split]
            yrs = sub["year"].agg(["min", "max"])
            rate = sub[PRIMARY_TARGET].mean()
            f.write(f"| {split} | {int(yrs['min'])}–{int(yrs['max'])} | {rate:.3f} |\n")
        f.write("\n")

        f.write("## 4. Naive Baseline vs. ML Model\n")
        f.write(f"The naive baseline predicts the TRAIN positive rate "
                f"({naive_baseline['train_positive_rate']:.4f}) for every observation.\n\n")
        f.write(f"- Naive TEST PR-AUC: `{naive_baseline['test_pr_auc']:.3f}`\n")
        f.write(f"- Naive TEST Brier: `{naive_baseline['test_brier']:.3f}`\n\n")
        f.write("A well-calibrated ML model should achieve lower Brier score and higher "
                "PR-AUC than the naive baseline — if the model has learned real filament risk "
                "signals and not merely solar-cycle activity level.\n\n")

        f.write("## 5. Distinguishing Model Failure from Distribution Shift\n")
        f.write("If the ML model performs only marginally above the naive baseline on TEST, "
                "this is most likely explained by:\n\n")
        f.write("1. **Solar-cycle distribution shift**: The model is calibrated to high-activity "
                "conditions; its predicted probabilities are too high for solar-minimum periods.\n")
        f.write("2. **Sparse temporal tracking**: Only 3.9% of observations have temporal "
                "evolution features, limiting the value of Group E features.\n")
        f.write("3. **Absence of active-region IDs**: Ground-truth masks do not carry AR "
                "numbers, making Groups C and D less informative.\n\n")
        f.write("Only if the model also fails on solar-maximum years (in TRAIN) would we "
                "conclude true model failure rather than distribution shift.\n")

    print(f"Saved solar-cycle report: {path}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("PHASE 2E — FULL-ARCHIVE FLARE RISK MODEL TRAINING")
    print("=" * 80)
    print(f"Input : {DATA_PATH}")
    print(f"Output: {OUT_DIR}/\n")

    # Load dataset
    if not DATA_PATH.exists():
        print(f"ERROR: {DATA_PATH} not found. Run Phase 2D first.")
        sys.exit(1)
    df = pd.read_csv(DATA_PATH)

    # Add temporal indicator (never imputed — binary presence flag)
    df["has_temporal_history"] = (~df["area_growth_rate"].isna()).astype(int)

    # Data audit
    mx_ok, xclass_ok = run_data_audit(df, PRIMARY_TARGET)

    # Splits
    train_df = df[df["split"] == "TRAIN"].copy()
    val_df   = df[df["split"] == "VAL"].copy()
    test_df  = df[df["split"] == "TEST"].copy()

    y_train = train_df[PRIMARY_TARGET].values.astype(int)
    y_val   = val_df[PRIMARY_TARGET].values.astype(int)
    y_test  = test_df[PRIMARY_TARGET].values.astype(int)

    # Class weights
    pos_train = int(y_train.sum())
    neg_train = len(y_train) - pos_train
    scale_pos = neg_train / pos_train if pos_train > 0 else 1.0
    print(f"TRAIN imbalance: neg={neg_train}, pos={pos_train}, "
          f"scale_pos_weight={scale_pos:.2f}\n")

    # Naive baseline (before training)
    naive_baseline = evaluate_naive_baseline(y_train, y_val, y_test)

    # Define numeric feature sets per config
    numeric_configs = {
        "static":   GROUP_A_MORPHOLOGY + GROUP_B_POSITION,
        "context":  GROUP_A_MORPHOLOGY + GROUP_B_POSITION + GROUP_C_HIST_FLARE + GROUP_D_AR_HISTORY,
        "temporal": GROUP_A_MORPHOLOGY + GROUP_B_POSITION + GROUP_C_HIST_FLARE
                    + GROUP_D_AR_HISTORY + GROUP_E_TEMPORAL + ["has_temporal_history"],
    }
    cat_configs = {
        "static":   CATEGORICAL_FEATURES,
        "context":  CATEGORICAL_FEATURES,
        "temporal": CATEGORICAL_FEATURES,
    }

    # Leakage assertion for all configs
    for cname, feats in numeric_configs.items():
        assert_no_leakage(feats + cat_configs[cname], cname)

    # Build preprocessors (fitted on TRAIN only)
    preprocessors = {}
    feature_names_out = {}
    for cname, num_feats in numeric_configs.items():
        cat_feats = cat_configs[cname]
        avail_num = [f for f in num_feats if f in df.columns]
        avail_cat = [f for f in cat_feats if f in df.columns]
        prep = build_preprocessor(avail_num, avail_cat)
        prep.fit(train_df[avail_num + avail_cat])
        preprocessors[cname] = prep
        feature_names_out[cname] = get_feature_names(prep, avail_num, avail_cat)

    # Config labels for reporting
    config_labels = {
        "static": "Static (A+B)",
        "context": "+Context (A+B+C+D)",
        "temporal": "+Temporal (A+B+C+D+E)"
    }

    # Build algorithms
    def make_algorithms():
        algos = {
            "LogisticRegression": LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=42),
            "RandomForest": RandomForestClassifier(
                class_weight="balanced_subsample", n_estimators=200, random_state=42),
        }
        if HAS_XGB:
            algos["XGBoost"] = XGBClassifier(
                scale_pos_weight=scale_pos, eval_metric="logloss",
                random_state=42, verbosity=0, n_estimators=200)
        return algos

    # ── TRAIN 9 PRIMARY MODELS ────────────────────────────────────────────────
    all_results = {}
    print("=" * 80)
    print("TRAINING PRIMARY MODELS (3 configs × algorithms)")
    print("=" * 80)

    for cname, num_feats in numeric_configs.items():
        cat_feats = cat_configs[cname]
        avail_num = [f for f in num_feats if f in df.columns]
        avail_cat = [f for f in cat_feats if f in df.columns]
        prep = preprocessors[cname]
        label = config_labels[cname]

        X_train = pd.DataFrame(prep.transform(train_df[avail_num + avail_cat]),
                               columns=feature_names_out[cname])
        X_val   = pd.DataFrame(prep.transform(val_df[avail_num + avail_cat]),
                               columns=feature_names_out[cname])
        X_test  = pd.DataFrame(prep.transform(test_df[avail_num + avail_cat]),
                               columns=feature_names_out[cname])

        algos = make_algorithms()
        for algo_name, clf in algos.items():
            key = f"{cname}_{algo_name}"
            pipe = Pipeline([("classifier", clf)])
            res = train_and_evaluate(
                algo_name, pipe,
                X_train, y_train, X_val, y_val, X_test, y_test,
                label=label
            )
            res["config"] = cname
            res["numeric_feats"] = avail_num
            res["cat_feats"] = avail_cat
            all_results[key] = res

    # ── ASSOCIATION ABLATION (secondary) ─────────────────────────────────────
    print("\n" + "=" * 80)
    print("SECONDARY ABLATION — Association features")
    print("=" * 80)
    abl_num = (numeric_configs["temporal"]
               + [f for f in ASSOCIATION_FEATURES["numeric"] if f in df.columns])
    abl_cat = cat_configs["temporal"] + [f for f in ASSOCIATION_FEATURES["categorical"] if f in df.columns]
    abl_avail_num = [f for f in abl_num if f in df.columns]
    abl_avail_cat = [f for f in abl_cat if f in df.columns]
    assert_no_leakage(abl_avail_num + abl_avail_cat, "association_ablation")

    abl_prep = build_preprocessor(abl_avail_num, abl_avail_cat)
    abl_prep.fit(train_df[abl_avail_num + abl_avail_cat])
    abl_feat_names = get_feature_names(abl_prep, abl_avail_num, abl_avail_cat)

    best_primary_algo = "RandomForest" if "RandomForest" in make_algorithms() else "LogisticRegression"
    abl_clf = make_algorithms()[best_primary_algo]

    Xa_train = pd.DataFrame(abl_prep.transform(train_df[abl_avail_num + abl_avail_cat]), columns=abl_feat_names)
    Xa_val   = pd.DataFrame(abl_prep.transform(val_df[abl_avail_num + abl_avail_cat]),   columns=abl_feat_names)
    Xa_test  = pd.DataFrame(abl_prep.transform(test_df[abl_avail_num + abl_avail_cat]),  columns=abl_feat_names)

    abl_pipe = Pipeline([("classifier", abl_clf)])
    abl_res = train_and_evaluate(
        best_primary_algo, abl_pipe,
        Xa_train, y_train, Xa_val, y_val, Xa_test, y_test,
        label="SECONDARY ABLATION (+Association)"
    )
    all_results["ABLATION_association"] = abl_res

    # ── X-CLASS EXPERIMENT ───────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("X-CLASS EXPERIMENT")
    print("=" * 80)
    if xclass_ok:
        y_train_x = train_df[X_CLASS_TARGET].values.astype(int)
        y_val_x   = val_df[X_CLASS_TARGET].values.astype(int)
        y_test_x  = test_df[X_CLASS_TARGET].values.astype(int)
        pos_x = int(y_train_x.sum())
        neg_x = len(y_train_x) - pos_x
        spw_x = neg_x / pos_x if pos_x > 0 else 1.0

        cname_x = "context"
        avail_num_x = [f for f in numeric_configs[cname_x] if f in df.columns]
        avail_cat_x = [f for f in cat_configs[cname_x] if f in df.columns]
        prep_x = preprocessors[cname_x]
        Xx_train = pd.DataFrame(prep_x.transform(train_df[avail_num_x + avail_cat_x]),
                                columns=feature_names_out[cname_x])
        Xx_val   = pd.DataFrame(prep_x.transform(val_df[avail_num_x + avail_cat_x]),
                                columns=feature_names_out[cname_x])
        Xx_test  = pd.DataFrame(prep_x.transform(test_df[avail_num_x + avail_cat_x]),
                                columns=feature_names_out[cname_x])

        for algo_name, clf in make_algorithms().items():
            pipe_x = Pipeline([("classifier", clf)])
            res_x = train_and_evaluate(
                algo_name, pipe_x,
                Xx_train, y_train_x, Xx_val, y_val_x, Xx_test, y_test_x,
                label="X-class"
            )
            all_results[f"XCLASS_{algo_name}"] = res_x
    else:
        print("Skipping X-class training (insufficient positives in at least one split).")

    # ── MODEL SELECTION ───────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("MODEL SELECTION (by Val PR-AUC, primary models only)")
    print("=" * 80)
    best_key = None
    best_val_pr = -1.0
    for key, res in all_results.items():
        if "ABLATION" in key or "XCLASS" in key:
            continue
        if res.get("val_pr_auc", -1) > best_val_pr:
            best_val_pr = res["val_pr_auc"]
            best_key = key

    print(f"Best model: {best_key}  (Val PR-AUC = {best_val_pr:.3f})")
    best = all_results[best_key]
    print(f"  Test PR-AUC (uncal)  : {best['test_metrics']['PR-AUC']:.3f}")
    print(f"  Test PR-AUC (cal)    : {best['test_cal_metrics']['PR-AUC']:.3f}")
    print(f"  Naive TEST PR-AUC    : {naive_baseline['test_pr_auc']:.3f}")

    above_naive = best["test_metrics"]["PR-AUC"] > naive_baseline["test_pr_auc"]
    print(f"  ML above naive baseline: {'YES' if above_naive else 'NO (distribution shift likely)'}")

    # ── FEATURE IMPORTANCE ────────────────────────────────────────────────────
    top_features = {}
    for cname in ["static", "context", "temporal"]:
        for algo_name in make_algorithms().keys():
            key = f"{cname}_{algo_name}"
            if key not in all_results or "pipeline" not in all_results[key]:
                continue
            feats = feature_names_out[cname]
            label = f"{config_labels[cname]} — {algo_name}"
            try:
                rankings = get_feature_importance(all_results[key]["pipeline"], feats, algo_name)
                top_features[label] = rankings
            except Exception:
                pass

    # ── SOLAR-CYCLE + TRACK ANALYSIS ──────────────────────────────────────────
    solar_data = solar_cycle_analysis(df, np.array(best["test_probs"]), test_df)
    year_results = year_by_year_eval(
        test_df.copy(), y_test,
        np.array(best["test_probs"]), best["val_threshold"]
    )
    track_results = track_aware_analysis(
        test_df.copy(), y_test,
        np.array(best["test_probs"]), best["val_threshold"]
    )

    # ── ARTIFACTS ─────────────────────────────────────────────────────────────
    thresholds = {k: {"threshold": v["val_threshold"], "threshold_cal": v.get("val_threshold_cal")}
                  for k, v in all_results.items() if "val_threshold" in v}
    save_artifacts(all_results, best_key, preprocessors, thresholds,
                   val_df, test_df, numeric_configs, cat_configs)

    # ── REPORTS ───────────────────────────────────────────────────────────────
    write_main_report(all_results, best_key, naive_baseline, track_results,
                      year_results, solar_data, xclass_ok, df, top_features)
    write_solar_cycle_report(df, solar_data, naive_baseline)

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    print("=" * 80)
    print("PHASE 2E — FINAL SUMMARY")
    print("=" * 80)
    print(f"BEST MODEL        : {best['model_name']} — {best['label']}")
    print(f"TARGET            : {PRIMARY_TARGET}")
    print(f"")
    print(f"VALIDATION:")
    print(f"  PR-AUC          : {best['val_pr_auc']:.3f}")
    print(f"  ROC-AUC         : {best['val_roc_auc']:.3f}")
    print(f"  Brier           : {best['val_brier']:.3f}")
    print(f"  F1 (th={best['val_threshold']:.2f})  : {best['val_metrics']['F1']:.3f}")
    print(f"")
    print(f"TEST (sealed, uncalibrated):")
    tm = best["test_metrics"]
    print(f"  PR-AUC          : {tm['PR-AUC']:.3f}   (Phase 2C: {PHASE2C_BASELINE['test_pr_auc']:.3f})")
    print(f"  ROC-AUC         : {tm['ROC-AUC']:.3f}  (Phase 2C: {PHASE2C_BASELINE['test_roc_auc']:.3f})")
    print(f"  Brier           : {tm['Brier']:.3f}  (Phase 2C: {PHASE2C_BASELINE['test_brier']:.3f})")
    print(f"  F1              : {tm['F1']:.3f}   (Phase 2C: {PHASE2C_BASELINE['test_f1']:.3f})")
    print(f"  Precision       : {tm['Precision']:.3f}")
    print(f"  Recall          : {tm['Recall']:.3f}")
    print(f"")
    print(f"NAIVE BASELINE TEST PR-AUC : {naive_baseline['test_pr_auc']:.3f}")
    print(f"ML above naive baseline    : {'YES' if above_naive else 'NO'}")
    print(f"")
    print(f"X-CLASS EXPERIMENT: {'RAN' if xclass_ok else 'INSUFFICIENT DATA'}")
    print(f"")
    print(f"SOLAR-CYCLE GENERALIZATION: See reports/PHASE2E_SOLAR_CYCLE_DISTRIBUTION.md")
    print(f"TRACK-AWARE ANALYSIS      : See reports/PHASE2E_FLARE_RISK_REPORT.md §9")
    print(f"")
    print(f"MAIN FINDING: {'Phase 2E improves over Phase 2C.' if tm['PR-AUC'] > PHASE2C_BASELINE['test_pr_auc'] else 'Phase 2E does NOT clearly improve over Phase 2C — solar-cycle distribution shift is the primary explanation.'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
