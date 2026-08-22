# PHASE 2E.1 — EVALUATION AUDIT

## 1. Class Probability Column
- Uncalibrated classes_: [0, 1]
- Calibrated classes_  : [0, 1]
## 2. Ranking Preservation
- Spearman correlation (raw vs cal): -1.0000
- Ranking inversions (first 1000 pairs): 55316

## 3. Metrics (Recalculated on exact same array)
- Raw PR-AUC: 0.2538
- Cal PR-AUC: 0.0296
- Raw ROC-AUC: 0.9059
- Cal ROC-AUC: 0.0941

## 4. Row Alignment
- Test target alignment: True
- Test raw prob alignment: True
- Test cal prob alignment: True

## 5. Calibration Procedure
- Calibrated Classifier: CalibratedClassifierCV (method='sigmoid', cv='prefit')
- Fitted on VAL set: 501 rows, 20 positive.

## 6. Brier & Log Loss
- Raw Brier: 0.0498
- Cal Brier: 0.0484
- Raw Log Loss: 0.1824
- Cal Log Loss: 0.2081

## 7. Naive Baseline
- Constant Train Rate: 0.1603
- Naive PR-AUC: 0.0503
- Naive ROC-AUC: 0.5000
- Naive Brier: 0.0599

## 9. Static Model Check
- Static Test PR-AUCs: [0.33503266 0.3112415  0.32699416]

## Conclusion
C: Evaluation implementation is invalid and Phase 2E test results must be recomputed.
