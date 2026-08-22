# Phase 2E — Full-Archive Filament → Flare Risk Model Report

Generated: 2026-08-22 21:53:40

## 1. Research Question
Can solar filament morphology, heliographic position, backward-looking solar flare history, active-region context, and temporal filament evolution predict whether an M-class or X-class solar flare will occur within 24 hours?

> **Primary target**: `M_X_WITHIN_24H`

## 2. Dataset Description
- **Source**: `data\training\filament_forecast_full.csv` (Phase 2D output)
- **Total observations**: 5238
- **Unique timestamps**: 707
- **Date range**: 2011–2022
- **M/X positives**: 742 (14.2%)
- **X-class positives**: 151

## 3. Solar-Cycle Distribution Shift
> [!WARNING]
> **Critical**: TRAIN covers solar maximum (higher M/X rate); VAL/TEST cover solar minimum. This is NOT a data processing error — it is real solar physics. Performance degradation from TRAIN→TEST must be attributed to this shift before concluding model failure.

| Split | Rows | M/X Positives | Positive Rate |
|---|---|---|---|
| TRAIN | 4399 | 705 | 16.0% |
| VAL | 501 | 20 | 4.0% |
| TEST | 338 | 17 | 5.0% |

## 4. Feature Groups
| Group | Features | Used In |
|---|---|---|
| A — Morphology | area, length, width, skeleton_length, aspect_ratio, sinuosity, orientation, confidence | All models |
| B — Solar Position | centroid_lat, centroid_lon, disk_position | All models |
| C — Historical Flare | recent_flare_count, recent_C_count, recent_M_count, recent_X_count, hours_since_previous_flare | +Context, +Temporal |
| D — Active Region | active_region_previous_flare_count, active_region_previous_M_count, active_region_previous_X_count | +Context, +Temporal |
| E — Temporal Evolution | area_growth_rate, length_growth_rate, width_growth_rate, centroid_velocity, orientation_change, aspect_ratio_change, area_acceleration, length_acceleration | +Temporal only |

**Notes**:
- `has_temporal_history` indicator added to +Temporal model (1 if predecessor existed, never imputed).
- `active_region` raw ID dropped — too high cardinality, no ordinal meaning.
- Association features (`best_assoc_score`, etc.) excluded from all primary models.

## 5. Naive Baseline (Solar Rate)
Predict `p = 0.1603` (TRAIN positive rate) for every observation.

| Split | Naive PR-AUC | Naive Brier |
|---|---|---|
| VAL | 0.040 | 0.053 |
| TEST | 0.050 | 0.060 |

ML models must exceed this baseline to demonstrate predictive value beyond simply knowing the ambient solar activity rate.

## 6. Model Comparison (Validation)
| Config | Algorithm | Val PR-AUC | Val ROC-AUC | Val Brier | Val F1 | Threshold |
|---|---|---|---|---|---|---|
| Static (A+B) | LogisticRegression | 0.033 | 0.389 | 0.171 | 0.078 | 0.11 |
| Static (A+B) | RandomForest | 0.043 | 0.501 | 0.054 | 0.094 | 0.15 |
| Static (A+B) | XGBoost | 0.043 | 0.523 | 0.082 | 0.094 | 0.09 |
| +Context (A+B+C+D) | LogisticRegression | 0.027 | 0.255 | 0.126 | 0.061 | 0.34 |
| +Context (A+B+C+D) | RandomForest | 0.044 | 0.526 | 0.048 | 0.096 | 0.06 |
| +Context (A+B+C+D) | XGBoost | 0.044 | 0.517 | 0.098 | 0.074 | 0.01 |
| +Temporal (A+B+C+D+E) | LogisticRegression | 0.027 | 0.253 | 0.131 | 0.060 | 0.33 |
| +Temporal (A+B+C+D+E) | RandomForest | 0.044 | 0.541 | 0.051 | 0.105 | 0.04 |
| +Temporal (A+B+C+D+E) | XGBoost | 0.040 | 0.467 | 0.095 | 0.076 | 0.22 |

## 7. Sealed Test Evaluation (Best Model)
**Best model**: `RandomForest` — `+Context (A+B+C+D)` (selected by Val PR-AUC = 0.044)

### Uncalibrated
| Metric | Value |
|---|---|
| PR-AUC | 0.254 |
| ROC-AUC | 0.906 |
| Brier | 0.050 |
| F1 | 0.191 |
| Precision | 0.106 |
| Recall | 1.000 |
| Specificity | 0.551 |
| BalancedAccuracy | 0.776 |

### Calibrated (Platt Sigmoid on VAL)
| Metric | Value |
|---|---|
| PR-AUC | 0.030 |
| ROC-AUC | 0.094 |
| Brier | 0.048 |
| F1 | 0.091 |
| Precision | 0.048 |
| Recall | 0.941 |
| Specificity | 0.009 |
| BalancedAccuracy | 0.475 |

#### Confusion Matrix
```
                 Pred Neg  Pred Pos
Actual Neg :     177       144      
Actual Pos :     0         17       
```

## 8. Feature Ablation Summary
| Feature Config | Best Val PR-AUC | Best Test PR-AUC |
|---|---|---|
| Static (A+B) | 0.043 | 0.311 |
| +Context (A+B+C+D) | 0.044 | 0.254 |
| +Temporal (A+B+C+D+E) | 0.044 | 0.218 |
| +Association (SECONDARY ABLATION) | 0.044 | 0.218 |

## 9. Track-Aware Analysis
### Singleton Filaments
- **n**: 320
- **mx_positive**: 17
- **PR-AUC**: 0.26792804308778767
- **Recall**: 1.0
- **Specificity**: 0.5445544554455446
- **F1**: 0.19767441860465115

### Tracked Filaments
- **n**: 18
- **skipped**: insufficient data

## 10. Year-by-Year Generalization (Test Period)
| Year | N | M/X Positive | Rate | Recall | Precision | PR-AUC |
|---|---|---|---|---|---|---|
| 2017 | 82 | 0 | 0.000 | N/A | N/A | N/A |
| 2018 | 41 | 0 | 0.000 | N/A | N/A | N/A |
| 2019 | 23 | 0 | 0.000 | N/A | N/A | N/A |
| 2020 | 19 | 0 | 0.000 | N/A | N/A | N/A |
| 2021 | 88 | 3 | 0.034 | 1.0 | 0.0625 | N/A |
| 2022 | 85 | 14 | 0.165 | 1.0 | 0.16470588235294117 | N/A |

## 11. Feature Explainability
> These are **model-associated predictors**, not causal relationships.

### Static (A+B) — LogisticRegression
01. `recent_max_flare_class_M1.3` (coefficient: -2.8683)
02. `recent_max_flare_class_M2.1` (coefficient: 2.4478)
03. `recent_max_flare_class_M2.0` (coefficient: -2.4467)
04. `recent_max_flare_class_M3.6` (coefficient: 2.4445)
05. `recent_max_flare_class_X2.0` (coefficient: 2.3866)
06. `recent_max_flare_class_M3.8` (coefficient: 2.3698)
07. `recent_max_flare_class_X2.1` (coefficient: -2.3452)
08. `recent_max_flare_class_M2.2` (coefficient: 2.3321)
09. `recent_max_flare_class_M1.9` (coefficient: 2.2075)
10. `recent_max_flare_class_M1.2` (coefficient: -2.1764)

### Static (A+B) — RandomForest
01. `width` (importance: 0.0944)
02. `centroid_lat` (importance: 0.0913)
03. `centroid_lon` (importance: 0.0888)
04. `sinuosity` (importance: 0.0887)
05. `aspect_ratio` (importance: 0.0868)
06. `disk_position` (importance: 0.0863)
07. `area` (importance: 0.0853)
08. `length` (importance: 0.0777)
09. `skeleton_length` (importance: 0.0762)
10. `recent_max_flare_class_M1.1` (importance: 0.0716)

### Static (A+B) — XGBoost
01. `recent_max_flare_class_M1.3` (importance: 0.0723)
02. `recent_max_flare_class_M1.1` (importance: 0.0631)
03. `recent_max_flare_class_M2.0` (importance: 0.0400)
04. `recent_max_flare_class_M1.2` (importance: 0.0385)
05. `recent_max_flare_class_X2.1` (importance: 0.0354)
06. `recent_max_flare_class_C2.1` (importance: 0.0343)
07. `recent_max_flare_class_M5.7` (importance: 0.0333)
08. `recent_max_flare_class_X6.9` (importance: 0.0315)
09. `recent_max_flare_class_M6.5` (importance: 0.0289)
10. `recent_max_flare_class_C8.4` (importance: 0.0279)

### +Context (A+B+C+D) — LogisticRegression
01. `recent_max_flare_class_C9.1` (coefficient: 3.4866)
02. `recent_max_flare_class_C3.4` (coefficient: 3.2618)
03. `recent_max_flare_class_M2.0` (coefficient: -2.9554)
04. `recent_max_flare_class_X4.9` (coefficient: 2.8819)
05. `recent_max_flare_class_M1.3` (coefficient: -2.7027)
06. `recent_max_flare_class_M1.9` (coefficient: 2.4013)
07. `recent_max_flare_class_M1.0` (coefficient: 2.3803)
08. `recent_max_flare_class_M3.4` (coefficient: 2.3191)
09. `recent_max_flare_class_X2.1` (coefficient: -2.0743)
10. `recent_max_flare_class_M1.7` (coefficient: 2.0097)

### +Context (A+B+C+D) — RandomForest
01. `hours_since_previous_flare` (importance: 0.3224)
02. `centroid_lon` (importance: 0.0542)
03. `width` (importance: 0.0536)
04. `centroid_lat` (importance: 0.0521)
05. `sinuosity` (importance: 0.0520)
06. `disk_position` (importance: 0.0508)
07. `aspect_ratio` (importance: 0.0495)
08. `recent_flare_count` (importance: 0.0491)
09. `area` (importance: 0.0491)
10. `skeleton_length` (importance: 0.0470)

### +Context (A+B+C+D) — XGBoost
01. `recent_max_flare_class_M1.3` (importance: 0.1422)
02. `recent_max_flare_class_M1.2` (importance: 0.0887)
03. `recent_max_flare_class_M5.7` (importance: 0.0815)
04. `recent_max_flare_class_M6.5` (importance: 0.0741)
05. `recent_max_flare_class_M1.4` (importance: 0.0580)
06. `recent_max_flare_class_X2.3` (importance: 0.0556)
07. `recent_max_flare_class_C2.1` (importance: 0.0480)
08. `recent_max_flare_class_M2.0` (importance: 0.0395)
09. `recent_max_flare_class_M1.9` (importance: 0.0339)
10. `recent_max_flare_class_M6.0` (importance: 0.0285)

### +Temporal (A+B+C+D+E) — LogisticRegression
01. `recent_max_flare_class_C9.1` (coefficient: 3.5370)
02. `recent_max_flare_class_C3.4` (coefficient: 3.3160)
03. `recent_max_flare_class_M2.0` (coefficient: -2.9638)
04. `recent_max_flare_class_X4.9` (coefficient: 2.8848)
05. `recent_max_flare_class_M1.3` (coefficient: -2.6576)
06. `recent_max_flare_class_M1.9` (coefficient: 2.4136)
07. `recent_max_flare_class_M1.0` (coefficient: 2.3913)
08. `recent_max_flare_class_M3.4` (coefficient: 2.3295)
09. `recent_max_flare_class_X2.1` (coefficient: -2.0617)
10. `recent_max_flare_class_M1.7` (coefficient: 2.0291)

### +Temporal (A+B+C+D+E) — RandomForest
01. `hours_since_previous_flare` (importance: 0.3414)
02. `width` (importance: 0.0504)
03. `centroid_lat` (importance: 0.0500)
04. `disk_position` (importance: 0.0494)
05. `sinuosity` (importance: 0.0488)
06. `centroid_lon` (importance: 0.0482)
07. `aspect_ratio` (importance: 0.0479)
08. `area` (importance: 0.0471)
09. `recent_flare_count` (importance: 0.0447)
10. `length` (importance: 0.0430)

### +Temporal (A+B+C+D+E) — XGBoost
01. `recent_max_flare_class_M1.3` (importance: 0.1377)
02. `recent_max_flare_class_M1.2` (importance: 0.0859)
03. `recent_max_flare_class_M5.7` (importance: 0.0789)
04. `recent_max_flare_class_M6.5` (importance: 0.0718)
05. `recent_max_flare_class_M1.4` (importance: 0.0538)
06. `recent_max_flare_class_X2.3` (importance: 0.0530)
07. `recent_max_flare_class_C2.8` (importance: 0.0420)
08. `recent_max_flare_class_M2.0` (importance: 0.0387)
09. `recent_max_flare_class_C2.1` (importance: 0.0322)
10. `recent_M_count` (importance: 0.0294)

## 12. X-Class Analysis
> [!NOTE]
> **X-class data available, but insufficient for reliable split-level evaluation** (at least one split had fewer than 5 X-class positive examples). No X-class classifier was trained.

## 13. Comparison with Phase 2C Baseline
| Metric | Phase 2C | Phase 2E |
|---|---:|---:|
| Dataset rows | 507 | 5238 |
| Unique timestamps | 40 | 707 |
| M/X positives | 135 | 742 |
| Test PR-AUC | 0.336 | 0.254 |
| Test ROC-AUC | 0.292 | 0.906 |
| Test Brier | 0.380 | 0.050 |
| Test F1 | 0.109 | 0.191 |

## 14. Limitations
1. The chronological split means TRAIN/TEST span different phases of the solar cycle. Performance on TEST may understate true skill during solar maximum.
2. Temporal tracking links only 3.9% of observations — most filaments have no measured predecessor; temporal evolution features are sparse.
3. Active-region IDs are unavailable in the ground-truth annotations; the AR history features default to zero for most observations.
4. Do NOT claim operational forecasting capability based on this experiment.

## 15. Final Model Recommendation
**Selected model**: `RandomForest` — `+Context (A+B+C+D)`

- Validation PR-AUC: 0.044
- Test PR-AUC (uncalibrated): 0.254
- Test PR-AUC (calibrated): 0.030
- Frozen threshold: 0.06

