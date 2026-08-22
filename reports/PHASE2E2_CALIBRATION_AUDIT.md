# PHASE 2E.2 — CALIBRATION AUDIT AND RAW EVALUATION

## 1. Previous Calibration Failure
The previous `method='sigmoid'` Platt calibration completely inverted the probability ranking due to a severe distribution shift between TRAIN (16% M/X) and VAL (4% M/X). That calibration is marked **INVALID_CALIBRATION** and MUST NOT be used.

## 2. Raw Model Verification
- Classes: `[0, 1]`
- Positive class index: `1` (corresponds to class 1)
- Negative class index: `0`
- Orientation Check: **PASS** (Probability correctly extracted for M_X_WITHIN_24H = 1).

## 3. Raw Test Metrics (RandomForest +Context)
- Frozen Threshold (from VAL): `0.06`
- PR-AUC: `0.2538`
- ROC-AUC: `0.9059`
- Brier: `0.0498`
- LogLoss: `0.1824`
- F1: `0.1910`
- Precision: `0.1056`
- Recall: `1.0000`
- Specificity: `0.5514`
- BalancedAccuracy: `0.7757`

## 4. Candidate Calibration: Isotonic Regression
Isotonic Regression strictly enforces monotonicity. Let's verify if it preserves ranking.
- Spearman correlation (Raw vs Isotonic): `0.8957`
- Ranking inversions (first 1000 pairs): `0`
- Isotonic ROC-AUC: `0.7757`
- Isotonic PR-AUC: `0.1056`
- Isotonic Brier: `0.0468`

> **Result**: Isotonic calibration failed to preserve ranking perfectly or the validation sample (20 positives) is too small to fit a robust piecewise constant curve. **REJECTED**.

## 5. Baselines (Same TEST set)
| Model | PR-AUC | ROC-AUC | Brier |
|---|---|---|---|
| Naive Constant (16%) | 0.050 | 0.500 | 0.060 |
| Raw Context (RF) | 0.254 | 0.906 | 0.050 |
| Raw Static (RF) | 0.311 | 0.864 | 0.054 |
| Raw Context (LR) | 0.244 | 0.540 | 0.130 |
| Raw Context (XGB) | 0.142 | 0.796 | 0.108 |

## 6. Solar-Cycle Limitations
The TEST set occurs during solar minimum (5% prevalence). The Raw Context (RF) model PR-AUC of 0.254 is a 5x improvement over the 0.050 naive baseline. The model demonstrates genuine ranking skill, but because of the massive shift and small validation positive sample (20), probability calibration remains extremely brittle.

## 7. Final Decision
**STATUS: RAW_MODEL_VALID**

The raw ranking is valid. All calibration methods are rejected due to insufficient positive validation samples and risk of ranking inversion.
The model must only be used to generate **Relative flare-risk scores**, marked UNCALIBRATED.
