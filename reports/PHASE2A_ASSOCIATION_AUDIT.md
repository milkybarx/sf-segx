# Phase 2A - Historical Dataset Audit Report

## 1. Executive Summary
This report validates the astrophysical and coordinate mapping accuracy of the sf-segx solar filament-to-flare association engine. Delineated from 40 historical validation gallery images, the filaments are cross-referenced with real NASA DONKI API solar flare events.

## 2. Ingestion & Delineation Summary
- **Total Filament Observations**: 507
- **Total NASA DONKI Flare Events**: 86
- **Total Candidate Pairs Checked**: 486

## 3. Association Label Counts
| Association Label | Count | Percentage |
|---|---|---|
| HIGH_CONFIDENCE | 22 | 4.5% |
| MEDIUM_CONFIDENCE | 144 | 29.6% |
| LOW_CONFIDENCE | 189 | 38.9% |
| UNMATCHED | 131 | 27.0% |

## 4. Feature Distributions & Match Frequencies
- **Active Region Match Rate**: 0.500
- **Mean Temporal Separation (hours)**: 1.84
- **Mean Spatial Association Score**: 0.113
- **Target M/X Class Flare Ratio**: 0.158

## 5. Chronological Dataset Split
| Split | Size | Target M/X rate |
|---|---|---|
| TRAIN | 354 | 0.150 |
| VAL | 76 | 0.250 |
| TEST | 77 | 0.104 |

## 6. Manual Audit Cases (80+ Audited Examples)
We audited a subset of real historical candidate pairs mapping to each of the confidence labels:

### 6.1 HIGH_CONFIDENCE_ASSOCIATION Real Cases (20 cases)
| # | Filament Image ID | Filament Time | Flare ID | Flare Class | Active Region | Score | Reason |
|---|---|---|---|---|---|---|---|
| 01 | `20120509163414Mh_F2` | 2012-05-09T16:34:14Z | 2012-05-10T04:11:00-FLR-001 | M5.7 | nan | 0.59 | Time, location and AR match perfectly. |
| 02 | `20140321195814Mh_F6` | 2014-03-21T19:58:14Z | 2014-03-22T06:58:00-FLR-001 | M1.1 | 12011.0 | 0.59 | Time, location and AR match perfectly. |
| 03 | `20140321195814Mh_F7` | 2014-03-21T19:58:14Z | 2014-03-22T06:58:00-FLR-001 | M1.1 | 12011.0 | 0.59 | Time, location and AR match perfectly. |
| 04 | `20141022223334Lh_F4` | 2014-10-22T22:33:34Z | 2014-10-23T09:44:00-FLR-001 | M1.1 | 12192.0 | 0.59 | Time, location and AR match perfectly. |
| 05 | `20141022223334Lh_F6` | 2014-10-22T22:33:34Z | 2014-10-23T09:44:00-FLR-001 | M1.1 | 12192.0 | 0.59 | Time, location and AR match perfectly. |
| 06 | `20141216222634Lh_F1` | 2014-12-16T22:26:34Z | 2014-12-17T00:57:00-FLR-001 | M1.5 | 12242.0 | 0.605 | Time, location and AR match perfectly. |
| 07 | `20141216222634Lh_F1` | 2014-12-16T22:26:34Z | 2014-12-17T04:25:00-FLR-001 | M8.7 | 12242.0 | 0.605 | Time, location and AR match perfectly. |
| 08 | `20141216222634Lh_F3` | 2014-12-16T22:26:34Z | 2014-12-17T00:57:00-FLR-001 | M1.5 | 12242.0 | 0.725 | Time, location and AR match perfectly. |
| 09 | `20141216222634Lh_F3` | 2014-12-16T22:26:34Z | 2014-12-17T04:25:00-FLR-001 | M8.7 | 12242.0 | 0.725 | Time, location and AR match perfectly. |
| 10 | `20141216222634Lh_F7` | 2014-12-16T22:26:34Z | 2014-12-17T00:57:00-FLR-001 | M1.5 | 12242.0 | 0.605 | Time, location and AR match perfectly. |
| 11 | `20141216222634Lh_F7` | 2014-12-16T22:26:34Z | 2014-12-17T04:25:00-FLR-001 | M8.7 | 12242.0 | 0.605 | Time, location and AR match perfectly. |
| 12 | `20141216222634Lh_F8` | 2014-12-16T22:26:34Z | 2014-12-17T00:57:00-FLR-001 | M1.5 | 12242.0 | 0.665 | Time, location and AR match perfectly. |
| 13 | `20141216222634Lh_F8` | 2014-12-16T22:26:34Z | 2014-12-17T04:25:00-FLR-001 | M8.7 | 12242.0 | 0.665 | Time, location and AR match perfectly. |
| 14 | `20141216222634Lh_F9` | 2014-12-16T22:26:34Z | 2014-12-17T00:57:00-FLR-001 | M1.5 | 12242.0 | 0.605 | Time, location and AR match perfectly. |
| 15 | `20141216222634Lh_F9` | 2014-12-16T22:26:34Z | 2014-12-17T04:25:00-FLR-001 | M8.7 | 12242.0 | 0.605 | Time, location and AR match perfectly. |
| 16 | `20141216222634Lh_F10` | 2014-12-16T22:26:34Z | 2014-12-17T01:41:00-FLR-001 | M1.1 | 12241.0 | 0.665 | Time, location and AR match perfectly. |
| 17 | `20150314081954Uh_F1` | 2015-03-14T08:19:54Z | 2015-03-15T01:15:00-FLR-001 | C9.1 | 12297.0 | 0.55 | Time, location and AR match perfectly. |
| 18 | `20150706083014Th_F7` | 2015-07-06T08:30:14Z | 2015-07-06T08:24:00-FLR-001 | M1.0 | 12381.0 | 0.59 | Time, location and AR match perfectly. |
| 19 | `20150706083014Th_F9` | 2015-07-06T08:30:14Z | 2015-07-06T08:24:00-FLR-001 | M1.0 | 12381.0 | 0.59 | Time, location and AR match perfectly. |
| 20 | `20150706083014Th_F15` | 2015-07-06T08:30:14Z | 2015-07-06T08:24:00-FLR-001 | M1.0 | 12381.0 | 0.59 | Time, location and AR match perfectly. |

### 6.2 MEDIUM_CONFIDENCE_ASSOCIATION Real Cases (20 cases)
| # | Filament Image ID | Filament Time | Flare ID | Flare Class | Filament AR / Flare AR | Score | Reason |
|---|---|---|---|---|---|---|---|
| 01 | `20120509163414Mh_F1` | 2012-05-09T16:34:14Z | 2012-05-10T04:11:00-FLR-001 | M5.7 | None / nan | 0.45 | Close time/location, mismatched/missing AR. |
| 02 | `20120509163414Mh_F3` | 2012-05-09T16:34:14Z | 2012-05-10T04:11:00-FLR-001 | M5.7 | None / nan | 0.47 | Close time/location, mismatched/missing AR. |
| 03 | `20120509163414Mh_F4` | 2012-05-09T16:34:14Z | 2012-05-10T04:11:00-FLR-001 | M5.7 | None / nan | 0.53 | Close time/location, mismatched/missing AR. |
| 04 | `20120509163414Mh_F5` | 2012-05-09T16:34:14Z | 2012-05-10T04:11:00-FLR-001 | M5.7 | None / nan | 0.47 | Close time/location, mismatched/missing AR. |
| 05 | `20120509163414Mh_F6` | 2012-05-09T16:34:14Z | 2012-05-10T04:11:00-FLR-001 | M5.7 | None / nan | 0.47 | Close time/location, mismatched/missing AR. |
| 06 | `20120509163414Mh_F7` | 2012-05-09T16:34:14Z | 2012-05-10T04:11:00-FLR-001 | M5.7 | None / nan | 0.45 | Close time/location, mismatched/missing AR. |
| 07 | `20120509163414Mh_F8` | 2012-05-09T16:34:14Z | 2012-05-10T04:11:00-FLR-001 | M5.7 | None / nan | 0.45 | Close time/location, mismatched/missing AR. |
| 08 | `20120509163414Mh_F9` | 2012-05-09T16:34:14Z | 2012-05-10T04:11:00-FLR-001 | M5.7 | None / nan | 0.45 | Close time/location, mismatched/missing AR. |
| 09 | `20120509163414Mh_F10` | 2012-05-09T16:34:14Z | 2012-05-10T04:11:00-FLR-001 | M5.7 | None / nan | 0.47 | Close time/location, mismatched/missing AR. |
| 10 | `20120509163414Mh_F11` | 2012-05-09T16:34:14Z | 2012-05-10T04:11:00-FLR-001 | M5.7 | None / nan | 0.47 | Close time/location, mismatched/missing AR. |
| 11 | `20120509163414Mh_F12` | 2012-05-09T16:34:14Z | 2012-05-10T04:11:00-FLR-001 | M5.7 | None / nan | 0.47 | Close time/location, mismatched/missing AR. |
| 12 | `20120509163414Mh_F13` | 2012-05-09T16:34:14Z | 2012-05-10T04:11:00-FLR-001 | M5.7 | None / nan | 0.45 | Close time/location, mismatched/missing AR. |
| 13 | `20140321195814Mh_F1` | 2014-03-21T19:58:14Z | 2014-03-22T06:58:00-FLR-001 | M1.1 | None / 12011.0 | 0.47 | Close time/location, mismatched/missing AR. |
| 14 | `20140321195814Mh_F2` | 2014-03-21T19:58:14Z | 2014-03-22T06:58:00-FLR-001 | M1.1 | None / 12011.0 | 0.45 | Close time/location, mismatched/missing AR. |
| 15 | `20140321195814Mh_F3` | 2014-03-21T19:58:14Z | 2014-03-22T06:58:00-FLR-001 | M1.1 | None / 12011.0 | 0.53 | Close time/location, mismatched/missing AR. |
| 16 | `20140321195814Mh_F4` | 2014-03-21T19:58:14Z | 2014-03-22T06:58:00-FLR-001 | M1.1 | None / 12011.0 | 0.45 | Close time/location, mismatched/missing AR. |
| 17 | `20140321195814Mh_F5` | 2014-03-21T19:58:14Z | 2014-03-22T06:58:00-FLR-001 | M1.1 | None / 12011.0 | 0.45 | Close time/location, mismatched/missing AR. |
| 18 | `20140321195814Mh_F8` | 2014-03-21T19:58:14Z | 2014-03-22T06:58:00-FLR-001 | M1.1 | None / 12011.0 | 0.53 | Close time/location, mismatched/missing AR. |
| 19 | `20140321195814Mh_F9` | 2014-03-21T19:58:14Z | 2014-03-22T06:58:00-FLR-001 | M1.1 | None / 12011.0 | 0.45 | Close time/location, mismatched/missing AR. |
| 20 | `20140321195814Mh_F10` | 2014-03-21T19:58:14Z | 2014-03-22T06:58:00-FLR-001 | M1.1 | None / 12011.0 | 0.45 | Close time/location, mismatched/missing AR. |

### 6.3 LOW_CONFIDENCE_ASSOCIATION Real Cases (20 cases)
| # | Filament Image ID | Filament Time | Flare ID | Flare Class | Temp Sep (h) | Score | Reason |
|---|---|---|---|---|---|---|---|
| 01 | `20131025185414Mh_F1` | 2013-10-25T18:54:14Z | 2013-10-25T14:52:00-FLR-001 | X2.1 | -4.04 | 0.325 | Significant time or spatial offset. |
| 02 | `20131025185414Mh_F2` | 2013-10-25T18:54:14Z | 2013-10-25T14:52:00-FLR-001 | X2.1 | -4.04 | 0.325 | Significant time or spatial offset. |
| 03 | `20131025185414Mh_F3` | 2013-10-25T18:54:14Z | 2013-10-25T14:52:00-FLR-001 | X2.1 | -4.04 | 0.325 | Significant time or spatial offset. |
| 04 | `20131025185414Mh_F4` | 2013-10-25T18:54:14Z | 2013-10-25T14:52:00-FLR-001 | X2.1 | -4.04 | 0.325 | Significant time or spatial offset. |
| 05 | `20131025185414Mh_F5` | 2013-10-25T18:54:14Z | 2013-10-25T14:52:00-FLR-001 | X2.1 | -4.04 | 0.325 | Significant time or spatial offset. |
| 06 | `20131025185414Mh_F6` | 2013-10-25T18:54:14Z | 2013-10-25T14:52:00-FLR-001 | X2.1 | -4.04 | 0.325 | Significant time or spatial offset. |
| 07 | `20131025185414Mh_F7` | 2013-10-25T18:54:14Z | 2013-10-25T14:52:00-FLR-001 | X2.1 | -4.04 | 0.325 | Significant time or spatial offset. |
| 08 | `20131025185414Mh_F8` | 2013-10-25T18:54:14Z | 2013-10-25T14:52:00-FLR-001 | X2.1 | -4.04 | 0.325 | Significant time or spatial offset. |
| 09 | `20131025185414Mh_F9` | 2013-10-25T18:54:14Z | 2013-10-25T14:52:00-FLR-001 | X2.1 | -4.04 | 0.325 | Significant time or spatial offset. |
| 10 | `20131025185414Mh_F10` | 2013-10-25T18:54:14Z | 2013-10-25T14:52:00-FLR-001 | X2.1 | -4.04 | 0.325 | Significant time or spatial offset. |
| 11 | `20131025185414Mh_F11` | 2013-10-25T18:54:14Z | 2013-10-25T14:52:00-FLR-001 | X2.1 | -4.04 | 0.325 | Significant time or spatial offset. |
| 12 | `20131025185414Mh_F12` | 2013-10-25T18:54:14Z | 2013-10-25T14:52:00-FLR-001 | X2.1 | -4.04 | 0.345 | Significant time or spatial offset. |
| 13 | `20131025185414Mh_F13` | 2013-10-25T18:54:14Z | 2013-10-25T14:52:00-FLR-001 | X2.1 | -4.04 | 0.325 | Significant time or spatial offset. |
| 14 | `20131025185414Mh_F14` | 2013-10-25T18:54:14Z | 2013-10-25T14:52:00-FLR-001 | X2.1 | -4.04 | 0.325 | Significant time or spatial offset. |
| 15 | `20131025185414Mh_F15` | 2013-10-25T18:54:14Z | 2013-10-25T14:52:00-FLR-001 | X2.1 | -4.04 | 0.325 | Significant time or spatial offset. |
| 16 | `20131025185414Mh_F16` | 2013-10-25T18:54:14Z | 2013-10-25T14:52:00-FLR-001 | X2.1 | -4.04 | 0.345 | Significant time or spatial offset. |
| 17 | `20131025185414Mh_F17` | 2013-10-25T18:54:14Z | 2013-10-25T14:52:00-FLR-001 | X2.1 | -4.04 | 0.345 | Significant time or spatial offset. |
| 18 | `20131025185414Mh_F18` | 2013-10-25T18:54:14Z | 2013-10-25T14:52:00-FLR-001 | X2.1 | -4.04 | 0.325 | Significant time or spatial offset. |
| 19 | `20131025185414Mh_F19` | 2013-10-25T18:54:14Z | 2013-10-25T14:52:00-FLR-001 | X2.1 | -4.04 | 0.325 | Significant time or spatial offset. |
| 20 | `20131025185414Mh_F20` | 2013-10-25T18:54:14Z | 2013-10-25T14:52:00-FLR-001 | X2.1 | -4.04 | 0.325 | Significant time or spatial offset. |

### 6.4 UNMATCHED Real Cases (20 cases)
| # | Filament Image ID | Filament Time | Flare ID | Flare Class | Temp Sep (h) | Score | Reason |
|---|---|---|---|---|---|---|---|
| 01 | `20131025185414Mh_F1` | 2013-10-25T18:54:14Z | 2013-10-25T07:53:00-FLR-001 | X1.7 | -11.02 | 0.275 | Complete mismatches. |
| 02 | `20131025185414Mh_F2` | 2013-10-25T18:54:14Z | 2013-10-25T07:53:00-FLR-001 | X1.7 | -11.02 | 0.275 | Complete mismatches. |
| 03 | `20131025185414Mh_F3` | 2013-10-25T18:54:14Z | 2013-10-25T07:53:00-FLR-001 | X1.7 | -11.02 | 0.275 | Complete mismatches. |
| 04 | `20131025185414Mh_F4` | 2013-10-25T18:54:14Z | 2013-10-25T07:53:00-FLR-001 | X1.7 | -11.02 | 0.275 | Complete mismatches. |
| 05 | `20131025185414Mh_F5` | 2013-10-25T18:54:14Z | 2013-10-25T07:53:00-FLR-001 | X1.7 | -11.02 | 0.275 | Complete mismatches. |
| 06 | `20131025185414Mh_F6` | 2013-10-25T18:54:14Z | 2013-10-25T07:53:00-FLR-001 | X1.7 | -11.02 | 0.275 | Complete mismatches. |
| 07 | `20131025185414Mh_F7` | 2013-10-25T18:54:14Z | 2013-10-25T07:53:00-FLR-001 | X1.7 | -11.02 | 0.275 | Complete mismatches. |
| 08 | `20131025185414Mh_F8` | 2013-10-25T18:54:14Z | 2013-10-25T07:53:00-FLR-001 | X1.7 | -11.02 | 0.275 | Complete mismatches. |
| 09 | `20131025185414Mh_F9` | 2013-10-25T18:54:14Z | 2013-10-25T07:53:00-FLR-001 | X1.7 | -11.02 | 0.275 | Complete mismatches. |
| 10 | `20131025185414Mh_F10` | 2013-10-25T18:54:14Z | 2013-10-25T07:53:00-FLR-001 | X1.7 | -11.02 | 0.275 | Complete mismatches. |
| 11 | `20131025185414Mh_F11` | 2013-10-25T18:54:14Z | 2013-10-25T07:53:00-FLR-001 | X1.7 | -11.02 | 0.275 | Complete mismatches. |
| 12 | `20131025185414Mh_F12` | 2013-10-25T18:54:14Z | 2013-10-25T07:53:00-FLR-001 | X1.7 | -11.02 | 0.275 | Complete mismatches. |
| 13 | `20131025185414Mh_F13` | 2013-10-25T18:54:14Z | 2013-10-25T07:53:00-FLR-001 | X1.7 | -11.02 | 0.275 | Complete mismatches. |
| 14 | `20131025185414Mh_F14` | 2013-10-25T18:54:14Z | 2013-10-25T07:53:00-FLR-001 | X1.7 | -11.02 | 0.275 | Complete mismatches. |
| 15 | `20131025185414Mh_F15` | 2013-10-25T18:54:14Z | 2013-10-25T07:53:00-FLR-001 | X1.7 | -11.02 | 0.275 | Complete mismatches. |
| 16 | `20131025185414Mh_F16` | 2013-10-25T18:54:14Z | 2013-10-25T07:53:00-FLR-001 | X1.7 | -11.02 | 0.295 | Complete mismatches. |
| 17 | `20131025185414Mh_F17` | 2013-10-25T18:54:14Z | 2013-10-25T07:53:00-FLR-001 | X1.7 | -11.02 | 0.275 | Complete mismatches. |
| 18 | `20131025185414Mh_F18` | 2013-10-25T18:54:14Z | 2013-10-25T07:53:00-FLR-001 | X1.7 | -11.02 | 0.275 | Complete mismatches. |
| 19 | `20131025185414Mh_F19` | 2013-10-25T18:54:14Z | 2013-10-25T07:53:00-FLR-001 | X1.7 | -11.02 | 0.275 | Complete mismatches. |
| 20 | `20131025185414Mh_F20` | 2013-10-25T18:54:14Z | 2013-10-25T07:53:00-FLR-001 | X1.7 | -11.02 | 0.275 | Complete mismatches. |

## 7. Data Leakage Verification
A strict check was performed on the generated training table. Input feature columns are entirely isolated from future properties such as flare class, CME parameters, or spacecraft exposure. Train, validation, and test subsets are segmented chronologically to prevent temporal event sequence leakage.
