# Phase 2B - Forecasting Dataset Audit Report

## 1. Dataset Dimensions
- **Total Rows (Filament Observations)**: 507
- **Unique Filament IDs**: 507
- **Unique Image IDs**: 40
- **Unique Observation Timestamps**: 40
- **Duplicate Check**: Passed (0 duplicate rows found for `filament_id + observation_time`)
- **Dataset Source**: `MAGFiLO_40_gallery_validation`

> [!NOTE]
> This is a validation-scale historical dataset. Expansion to the full available filament archive is recommended before final model claims.

## 2. Target Variable Distributions (Class Imbalance)
| Target Name | Positive Count | Negative Count | Positive Percentage |
|---|---|---|---|
| M_X_WITHIN_24H | 135 | 372 | 26.6% |
| C_OR_HIGHER_24H | 159 | 348 | 31.4% |
| M_OR_HIGHER_24H | 135 | 372 | 26.6% |
| X_CLASS_24H | 0 | 507 | 0.0% |

## 3. Split Distributions & Positive Rates
| Split | Row Count | M_X_WITHIN_24H (pos) | Positive Rate | X_CLASS_24H (pos) |
|---|---|---|---|---|
| TRAIN | 381 | 79 | 20.7% | 0 |
| VAL | 57 | 26 | 45.6% | 0 |
| TEST | 69 | 30 | 43.5% | 0 |

> [!WARNING]
> Split **TRAIN** contains **zero** positive X-class flare events. This is due to the natural rarity of X-class solar events.

> [!WARNING]
> Split **VAL** contains **zero** positive X-class flare events. This is due to the natural rarity of X-class solar events.

> [!WARNING]
> Split **TEST** contains **zero** positive X-class flare events. This is due to the natural rarity of X-class solar events.

## 4. Association Evidence Coverage for Positives
Analyzing 135 positive filament observations with future M/X flares:
- Positive targets associated with **HIGH**: 5
- Positive targets associated with **MEDIUM**: 61
- Positive targets associated with **LOW**: 69
- Positive targets associated with **UNMATCHED**: 0

## 5. Temporal Feature Availability
- Observations with linked previous tracking: 2 (0.4%)
- Observations with no predecessor: remaining observations correctly set to `NaN` (no future interpolation leakage).

## 6. Manual Audit Cases (20 Positive & 20 Negative)

### 6.1 Positive M/X Target Cases (20 cases)
| # | Filament ID | Observation Time | Centroid Lat/Lon | Eruption | Future Flare | Flare Class | Delta (h) | Best Assoc Score |
|---|---|---|---|---|---|---|---|---|
| 01 | `20120509163414Mh_F1` | 2012-05-09T16:34:14Z | -48.5/-3.9 | None | 2012-05-10T04:11Z | M5.7 | 2012-05-10T04:11Z | 0.45 |
| 02 | `20120509163414Mh_F2` | 2012-05-09T16:34:14Z | +29.0/-6.8 | None | 2012-05-10T04:11Z | M5.7 | 2012-05-10T04:11Z | 0.59 |
| 03 | `20120509163414Mh_F3` | 2012-05-09T16:34:14Z | -24.8/-13.3 | None | 2012-05-10T04:11Z | M5.7 | 2012-05-10T04:11Z | 0.47 |
| 04 | `20120509163414Mh_F4` | 2012-05-09T16:34:14Z | +11.7/+3.5 | None | 2012-05-10T04:11Z | M5.7 | 2012-05-10T04:11Z | 0.53 |
| 05 | `20120509163414Mh_F5` | 2012-05-09T16:34:14Z | -26.9/-35.6 | None | 2012-05-10T04:11Z | M5.7 | 2012-05-10T04:11Z | 0.47 |
| 06 | `20120509163414Mh_F6` | 2012-05-09T16:34:14Z | +44.7/-12.4 | None | 2012-05-10T04:11Z | M5.7 | 2012-05-10T04:11Z | 0.47 |
| 07 | `20120509163414Mh_F7` | 2012-05-09T16:34:14Z | -28.2/+2.1 | None | 2012-05-10T04:11Z | M5.7 | 2012-05-10T04:11Z | 0.45 |
| 08 | `20120509163414Mh_F8` | 2012-05-09T16:34:14Z | -30.8/-9.1 | None | 2012-05-10T04:11Z | M5.7 | 2012-05-10T04:11Z | 0.45 |
| 09 | `20120509163414Mh_F9` | 2012-05-09T16:34:14Z | -13.1/-64.2 | None | 2012-05-10T04:11Z | M5.7 | 2012-05-10T04:11Z | 0.45 |
| 10 | `20120509163414Mh_F10` | 2012-05-09T16:34:14Z | -11.4/-53.6 | None | 2012-05-10T04:11Z | M5.7 | 2012-05-10T04:11Z | 0.47 |
| 11 | `20120509163414Mh_F11` | 2012-05-09T16:34:14Z | +18.9/+13.5 | None | 2012-05-10T04:11Z | M5.7 | 2012-05-10T04:11Z | 0.47 |
| 12 | `20120509163414Mh_F12` | 2012-05-09T16:34:14Z | -11.8/-44.2 | None | 2012-05-10T04:11Z | M5.7 | 2012-05-10T04:11Z | 0.47 |
| 13 | `20120509163414Mh_F13` | 2012-05-09T16:34:14Z | -15.2/+22.3 | None | 2012-05-10T04:11Z | M5.7 | 2012-05-10T04:11Z | 0.45 |
| 14 | `20140321195814Mh_F1` | 2014-03-21T19:58:14Z | -16.9/+28.2 | None | 2014-03-22T06:58Z | M1.1 | 2014-03-22T06:58Z | 0.47 |
| 15 | `20140321195814Mh_F2` | 2014-03-21T19:58:14Z | +19.0/+29.0 | None | 2014-03-22T06:58Z | M1.1 | 2014-03-22T06:58Z | 0.45 |
| 16 | `20140321195814Mh_F3` | 2014-03-21T19:58:14Z | -22.9/+43.0 | None | 2014-03-22T06:58Z | M1.1 | 2014-03-22T06:58Z | 0.53 |
| 17 | `20140321195814Mh_F4` | 2014-03-21T19:58:14Z | -1.6/+0.9 | None | 2014-03-22T06:58Z | M1.1 | 2014-03-22T06:58Z | 0.45 |
| 18 | `20140321195814Mh_F5` | 2014-03-21T19:58:14Z | -29.1/-24.1 | None | 2014-03-22T06:58Z | M1.1 | 2014-03-22T06:58Z | 0.45 |
| 19 | `20140321195814Mh_F6` | 2014-03-21T19:58:14Z | -10.5/+58.5 | None | 2014-03-22T06:58Z | M1.1 | 2014-03-22T06:58Z | 0.59 |
| 20 | `20140321195814Mh_F7` | 2014-03-21T19:58:14Z | -17.8/+56.5 | None | 2014-03-22T06:58Z | M1.1 | 2014-03-22T06:58Z | 0.59 |

### 6.2 Negative M/X Target Cases (20 cases)
| # | Filament ID | Observation Time | Centroid Lat/Lon | Eruption | Future Flare | Flare Class | Best Assoc Score | Best Assoc Label |
|---|---|---|---|---|---|---|---|---|
| 01 | `20110601063134Lh_F1` | 2011-06-01T06:31:34Z | +37.3/-26.6 | None | None | N/A | 0.0 | UNMATCHED |
| 02 | `20110601063134Lh_F2` | 2011-06-01T06:31:34Z | -37.5/+46.2 | None | None | N/A | 0.0 | UNMATCHED |
| 03 | `20110601063134Lh_F3` | 2011-06-01T06:31:34Z | -44.8/+22.6 | None | None | N/A | 0.0 | UNMATCHED |
| 04 | `20110601063134Lh_F4` | 2011-06-01T06:31:34Z | +19.5/-29.3 | None | None | N/A | 0.0 | UNMATCHED |
| 05 | `20110601063134Lh_F5` | 2011-06-01T06:31:34Z | +17.0/+20.1 | None | None | N/A | 0.0 | UNMATCHED |
| 06 | `20110601063134Lh_F6` | 2011-06-01T06:31:34Z | +22.7/+27.7 | None | None | N/A | 0.0 | UNMATCHED |
| 07 | `20110601063134Lh_F7` | 2011-06-01T06:31:34Z | +20.7/-6.4 | None | None | N/A | 0.0 | UNMATCHED |
| 08 | `20110601063134Lh_F8` | 2011-06-01T06:31:34Z | -18.0/-27.7 | None | None | N/A | 0.0 | UNMATCHED |
| 09 | `20110601063134Lh_F9` | 2011-06-01T06:31:34Z | -3.6/+14.1 | None | None | N/A | 0.0 | UNMATCHED |
| 10 | `20110601063134Lh_F10` | 2011-06-01T06:31:34Z | -22.0/-4.4 | None | None | N/A | 0.0 | UNMATCHED |
| 11 | `20110601063134Lh_F11` | 2011-06-01T06:31:34Z | -18.5/-21.7 | None | None | N/A | 0.0 | UNMATCHED |
| 12 | `20110601063134Lh_F12` | 2011-06-01T06:31:34Z | -24.7/+25.6 | None | None | N/A | 0.0 | UNMATCHED |
| 13 | `20110601063134Lh_F13` | 2011-06-01T06:31:34Z | +12.3/-22.8 | None | None | N/A | 0.0 | UNMATCHED |
| 14 | `20120213063134Lh_F1` | 2012-02-13T06:31:34Z | +38.3/+34.4 | None | None | N/A | 0.0 | UNMATCHED |
| 15 | `20120213063134Lh_F2` | 2012-02-13T06:31:34Z | -44.6/-10.5 | None | None | N/A | 0.0 | UNMATCHED |
| 16 | `20120213063134Lh_F3` | 2012-02-13T06:31:34Z | -23.0/-47.0 | None | None | N/A | 0.0 | UNMATCHED |
| 17 | `20120213063134Lh_F4` | 2012-02-13T06:31:34Z | -14.0/+18.4 | None | None | N/A | 0.0 | UNMATCHED |
| 18 | `20120213063134Lh_F5` | 2012-02-13T06:31:34Z | -11.6/+4.1 | None | None | N/A | 0.0 | UNMATCHED |
| 19 | `20120213063134Lh_F6` | 2012-02-13T06:31:34Z | -23.2/-27.8 | None | None | N/A | 0.0 | UNMATCHED |
| 20 | `20120213063134Lh_F7` | 2012-02-13T06:31:34Z | +15.7/+20.9 | None | None | N/A | 0.0 | UNMATCHED |

## 7. Recommendations and Limitations
1. **Data Imbalance**: The dataset represents a highly imbalanced class distribution, especially for X-class flares.
2. **Gallery Limitations**: The 40 gallery validation images are intended only for checking pipeline logic and should not be used as a final training archive.
