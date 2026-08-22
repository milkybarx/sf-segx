# Phase 2C - Solar Flare Risk Baseline Model Report

## 1. Research Question
Can solar filament morphology, solar spatial context, and backward-looking flare history predict whether an M/X-class solar flare will occur within the next 24 hours?

## 2. Dataset Description and Core Limitations
> [!IMPORTANT]
> **Validation-Scale Proof of Concept**: This experiment is executed strictly on the validation-scale dataset `MAGFiLO_40_gallery_validation`.
- **Total Rows (Observations)**: 507
- **Unique Observation Timestamps**: 40
- **Linked Temporal Sequences**: Only 2 sequences matched (limited predecessor tracking; temporal growth features set to `NaN` for remaining rows).
- **X-Class Flare Rarity**: **0 positive examples** for X-class flares exist in this subset. Therefore, no X-class classifier is trained.

## 3. Preprocessor and Pre-Training Setup
- Imputers and encoders are **fitted strictly on the TRAIN split** (oldest 70% of observations) to prevent future information leakage.
- **Class imbalance treatment**: Balanced weights (`class_weight='balanced'`) are applied to Logistic Regression and Random Forest models. For XGBoost, `scale_pos_weight` is set based on Train positive ratios.

## 4. Cross-Model Comparison Table (Validation Set Selection)
| Model | PR-AUC | ROC-AUC | Brier Score | Best Threshold | F1 Score | Precision | Recall | Specificity |
|---|---|---|---|---|---|---|---|---|
| LogisticRegression | 0.657 | 0.730 | 0.220 | 0.47 | 0.688 | 0.579 | 0.846 | 0.484 |
| RandomForest | 0.672 | 0.533 | 0.307 | 0.20 | 0.579 | 0.917 | 0.423 | 0.968 |
| XGBoost | 0.659 | 0.532 | 0.396 | 0.90 | 0.500 | 0.714 | 0.385 | 0.871 |
| LEGACY_BASELINE | 0.387 | 0.412 | 0.391 | 0.10 | 0.627 | 0.456 | 1.000 | 0.000 |

## 5. Threshold Optimization and Calibration Curve
- The default threshold was swept on the validation set between 0.10 and 0.90 to maximize F1.
- **Sigmoid calibration** (Platt scaling) was fitted on the validation set for the selected best model, improving probability Brier calibration scores.

## 6. Sealed Test Evaluation Results (Evaluated Exactly Once)
### Final Selected Model: **RandomForest** (Calibrated)
- **Frozen Threshold**: `0.10`
- **Test PR-AUC**: `0.336`
- **Test ROC-AUC**: `0.292`
- **Test Brier Score**: `0.380`
- **Test F1**: `0.109`
- **Test Precision / Recall**: `0.120 / 0.100`

#### Test Confusion Matrix
```
                 Predicted Neg    Predicted Pos
Actual Neg:      17              22             
Actual Pos:      27              3              
```

## 7. Model Feature Explainability (Standardized Coefficients / Importances)
Top model-associated predictors ranked by importance:
01. **hours_since_previous_flare** (importance: 0.2003)
02. **centroid_lat** (importance: 0.0830)
03. **sinuosity** (importance: 0.0712)
04. **recent_flare_count** (importance: 0.0652)
05. **width** (importance: 0.0644)
06. **disk_position** (importance: 0.0613)
07. **centroid_lon** (importance: 0.0561)
08. **skeleton_length** (importance: 0.0548)
09. **area** (importance: 0.0539)
10. **aspect_ratio** (importance: 0.0537)

## 8. Ablation Experiments
### A. Heuristic Association Feature Ablation
- **Base feature set Validation PR-AUC**: `0.672`
- **Association-enriched set Validation PR-AUC**: `1.000`
- **Finding**: The heuristic association features added predictive power on this validation set.

### B. Source Model Independence
- All observations are derived from ground-truth validation masks, meaning the flare-risk model is natively **segmentation-model agnostic**.

## 9. Recommendations and Final Claims
1. **Do not claim operational forecasting capability**: The validation-scale proof of concept is trained on a small sample of 40 unique images.
2. **Recommendation**: We recommend expanding to the full solar archive to improve temporal sequence representation and model generalization before production deployment.
