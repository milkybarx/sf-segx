# Phase 2E — Solar-Cycle Distribution Analysis

Generated: 2026-08-22 21:53:40

## 1. Background
Solar flare activity follows an approximately 11-year cycle. The Phase 2D dataset spans 2011–2022, which includes solar cycle 24 (peak ~2014) and the early rise of cycle 25. The chronological train/val/test split means TRAIN captures more active periods, while VAL/TEST capture the declining phase and solar minimum.

> [!IMPORTANT]
> Any model trained primarily on solar maximum data will likely see lower apparent precision in the VAL/TEST period. This is a **temporal distribution shift**, not necessarily model failure.

## 2. Year-by-Year M/X Positive Rate
| Year | N | M/X Positives | Positive Rate | Split |
|---|---|---|---|---|
| 2011 | 624 | 38 | 0.061 | TRAIN |
| 2012 | 806 | 76 | 0.094 | TRAIN |
| 2013 | 951 | 44 | 0.046 | TRAIN |
| 2014 | 1023 | 340 | 0.332 | TRAIN |
| 2015 | 792 | 205 | 0.259 | TRAIN |
| 2016 | 542 | 22 | 0.041 | VAL |
| 2017 | 244 | 0 | 0.000 | VAL |
| 2018 | 41 | 0 | 0.000 | TEST |
| 2019 | 23 | 0 | 0.000 | TEST |
| 2020 | 19 | 0 | 0.000 | TEST |
| 2021 | 88 | 3 | 0.034 | TEST |
| 2022 | 85 | 14 | 0.165 | TEST |

## 3. Train/Val/Test Regime Mapping
| Split | Years Covered | M/X Rate |
|---|---|---|
| TRAIN | 2011–2016 | 0.160 |
| VAL | 2016–2017 | 0.040 |
| TEST | 2017–2022 | 0.050 |

## 4. Naive Baseline vs. ML Model
The naive baseline predicts the TRAIN positive rate (0.1603) for every observation.

- Naive TEST PR-AUC: `0.050`
- Naive TEST Brier: `0.060`

A well-calibrated ML model should achieve lower Brier score and higher PR-AUC than the naive baseline — if the model has learned real filament risk signals and not merely solar-cycle activity level.

## 5. Distinguishing Model Failure from Distribution Shift
If the ML model performs only marginally above the naive baseline on TEST, this is most likely explained by:

1. **Solar-cycle distribution shift**: The model is calibrated to high-activity conditions; its predicted probabilities are too high for solar-minimum periods.
2. **Sparse temporal tracking**: Only 3.9% of observations have temporal evolution features, limiting the value of Group E features.
3. **Absence of active-region IDs**: Ground-truth masks do not carry AR numbers, making Groups C and D less informative.

Only if the model also fails on solar-maximum years (in TRAIN) would we conclude true model failure rather than distribution shift.
