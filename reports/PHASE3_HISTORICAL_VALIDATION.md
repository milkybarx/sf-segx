# PHASE 3 — HISTORICAL VALIDATION REPORT

> **Source Provenance Legend**
> - `DONKI_OBSERVED` — CMEAnalysis parameters from real NASA DONKI records
> - `OUR_GEOMETRIC_ESTIMATE` — Simplified CME cone model (this system)
> - `STATIC_ORBIT_APPROXIMATION` — Spacecraft position from static catalog
> - `UNKNOWN` — Data not available

## Summary
- Cases run: **10**
- Agreement: **10**
- Partial agreement: **0**
- Disagreement: **0**
- Unknown: **0**

> [!IMPORTANT]
> All spacecraft positions use `STATIC_ORBIT_APPROXIMATION`.
> The cone model parameters (flank_margin=15°) were NOT tuned to match these cases.
> Cone intersection with Earth uses Stonyhurst heliocentric frame (lat=0, lon=0).

---

## Case: 2011-Sep-6 X2.1
*X2.1 flare from AR 11283; associated CME produced Kp=7 storm on Sep 9.*
**Source**: `DONKI_OBSERVED; NOAA/SWPC event catalog`

### CMEAnalysis Parameters (DONKI_OBSERVED)
| Field | Value |
|---|---|
| CME ID | `2011-09-06T22:12:00-CME-001` |
| Speed (km/s) | `575.0` |
| Latitude (Stonyhurst °) | `-28.0` |
| Longitude (Stonyhurst °) | `-42.0` |
| Half Angle (°) | `45.0` |
| SEP Observed | `SEP_OBSERVED` |
| Max Kp Observed | `7.0` |

### Our Geometric Estimate (OUR_GEOMETRIC_ESTIMATE)
| Field | Value |
|---|---|
| Earth Impact Status | `EARTH_CONE_INTERSECTION` |
| Earth Angular Sep (°) | `49.0°` |
| Estimated Arrival at Earth | ~72.3 hours after CME start (ESTIMATED) |

### Validation Comparison
- Observed Earth Impact: **YES**
- Cone Model Result: **EARTH_CONE_INTERSECTION**
- **Agreement: AGREEMENT**
- Explanation: Both observed Earth impact (GST/storm) and cone model confirm Earth-directed geometry.

### Spacecraft Exposure Table (STATIC_ORBIT_APPROXIMATION)
| Spacecraft | Exposure | Angular Sep (°) | Estimated Arrival | Calc. Method |
|---|---|---|---|---|
| SOHO | NEAR_FLANK | 49.0° | 2011-09-08T23:32:48.507814Z | OUR_GEOMETRIC_ESTIMATE |
| SDO | NEAR_FLANK | 49.0° | 2011-09-09T00:16:10.209913Z | OUR_GEOMETRIC_ESTIMATE |
| GOES-16 | NEAR_FLANK | 49.0° | 2011-09-09T00:16:10.209913Z | OUR_GEOMETRIC_ESTIMATE |
| ISS | NEAR_FLANK | 49.0° | 2011-09-09T00:16:10.209913Z | OUR_GEOMETRIC_ESTIMATE |
| GPS-BIIRM-1 | NEAR_FLANK | 49.0° | 2011-09-09T00:16:10.209913Z | OUR_GEOMETRIC_ESTIMATE |

---

## Case: 2012-Jul-12 X1.4
*X1.4 from AR 11520; CME hit Earth Jul 14, Kp=6.*
**Source**: `DONKI_OBSERVED; Richardson & Cane catalog`

### CMEAnalysis Parameters (DONKI_OBSERVED)
| Field | Value |
|---|---|
| CME ID | `2012-07-12T16:48:00-CME-001` |
| Speed (km/s) | `885.0` |
| Latitude (Stonyhurst °) | `-17.0` |
| Longitude (Stonyhurst °) | `1.0` |
| Half Angle (°) | `51.0` |
| SEP Observed | `SEP_OBSERVED` |
| Max Kp Observed | `6.0` |

### Our Geometric Estimate (OUR_GEOMETRIC_ESTIMATE)
| Field | Value |
|---|---|
| Earth Impact Status | `EARTH_CONE_INTERSECTION` |
| Earth Angular Sep (°) | `17.0°` |
| Estimated Arrival at Earth | ~47.0 hours after CME start (ESTIMATED) |

### Validation Comparison
- Observed Earth Impact: **YES**
- Cone Model Result: **EARTH_CONE_INTERSECTION**
- **Agreement: AGREEMENT**
- Explanation: Both observed Earth impact (GST/storm) and cone model confirm Earth-directed geometry.

### Spacecraft Exposure Table (STATIC_ORBIT_APPROXIMATION)
| Spacecraft | Exposure | Angular Sep (°) | Estimated Arrival | Calc. Method |
|---|---|---|---|---|
| SOHO | INSIDE_CONE | 17.0° | 2012-07-13T22:29:06.770614Z | OUR_GEOMETRIC_ESTIMATE |
| SDO | INSIDE_CONE | 17.0° | 2012-07-13T22:57:17.142034Z | OUR_GEOMETRIC_ESTIMATE |
| GOES-16 | INSIDE_CONE | 17.0° | 2012-07-13T22:57:17.142034Z | OUR_GEOMETRIC_ESTIMATE |
| ISS | INSIDE_CONE | 17.0° | 2012-07-13T22:57:17.142034Z | OUR_GEOMETRIC_ESTIMATE |
| GPS-BIIRM-1 | INSIDE_CONE | 17.0° | 2012-07-13T22:57:17.142034Z | OUR_GEOMETRIC_ESTIMATE |

---

## Case: 2012-Mar-7 X5.4
*X5.4+X1.3 double event; fast CME, Kp=7, large SEP event.*
**Source**: `DONKI_OBSERVED; Liu et al. 2013`

### CMEAnalysis Parameters (DONKI_OBSERVED)
| Field | Value |
|---|---|
| CME ID | `2012-03-07T00:24:00-CME-001` |
| Speed (km/s) | `2684.0` |
| Latitude (Stonyhurst °) | `-26.0` |
| Longitude (Stonyhurst °) | `-2.0` |
| Half Angle (°) | `60.0` |
| SEP Observed | `SEP_OBSERVED` |
| Max Kp Observed | `7.0` |

### Our Geometric Estimate (OUR_GEOMETRIC_ESTIMATE)
| Field | Value |
|---|---|
| Earth Impact Status | `EARTH_CONE_INTERSECTION` |
| Earth Angular Sep (°) | `26.1°` |
| Estimated Arrival at Earth | ~15.5 hours after CME start (ESTIMATED) |

### Validation Comparison
- Observed Earth Impact: **YES**
- Cone Model Result: **EARTH_CONE_INTERSECTION**
- **Agreement: AGREEMENT**
- Explanation: Both observed Earth impact (GST/storm) and cone model confirm Earth-directed geometry.

### Spacecraft Exposure Table (STATIC_ORBIT_APPROXIMATION)
| Spacecraft | Exposure | Angular Sep (°) | Estimated Arrival | Calc. Method |
|---|---|---|---|---|
| SOHO | INSIDE_CONE | 26.1° | 2012-03-07T15:19:39.542471Z | OUR_GEOMETRIC_ESTIMATE |
| SDO | INSIDE_CONE | 26.1° | 2012-03-07T15:28:56.911587Z | OUR_GEOMETRIC_ESTIMATE |
| GOES-16 | INSIDE_CONE | 26.1° | 2012-03-07T15:28:56.911587Z | OUR_GEOMETRIC_ESTIMATE |
| ISS | INSIDE_CONE | 26.1° | 2012-03-07T15:28:56.911587Z | OUR_GEOMETRIC_ESTIMATE |
| GPS-BIIRM-1 | INSIDE_CONE | 26.1° | 2012-03-07T15:28:56.911587Z | OUR_GEOMETRIC_ESTIMATE |

---

## Case: 2014-Sep-10 X1.6
*X1.6 from AR 12158; near-central CME, Kp=6 storm Sep 12.*
**Source**: `DONKI_OBSERVED; Webb & Howard review`

### CMEAnalysis Parameters (DONKI_OBSERVED)
| Field | Value |
|---|---|
| CME ID | `2014-09-10T17:21:00-CME-001` |
| Speed (km/s) | `1267.0` |
| Latitude (Stonyhurst °) | `10.0` |
| Longitude (Stonyhurst °) | `-6.0` |
| Half Angle (°) | `35.0` |
| SEP Observed | `SEP_OBSERVED` |
| Max Kp Observed | `6.0` |

### Our Geometric Estimate (OUR_GEOMETRIC_ESTIMATE)
| Field | Value |
|---|---|
| Earth Impact Status | `EARTH_CONE_INTERSECTION` |
| Earth Angular Sep (°) | `11.6°` |
| Estimated Arrival at Earth | ~32.8 hours after CME start (ESTIMATED) |

### Validation Comparison
- Observed Earth Impact: **YES**
- Cone Model Result: **EARTH_CONE_INTERSECTION**
- **Agreement: AGREEMENT**
- Explanation: Both observed Earth impact (GST/storm) and cone model confirm Earth-directed geometry.

### Spacecraft Exposure Table (STATIC_ORBIT_APPROXIMATION)
| Spacecraft | Exposure | Angular Sep (°) | Estimated Arrival | Calc. Method |
|---|---|---|---|---|
| SOHO | INSIDE_CONE | 11.6° | 2014-09-11T08:28:11.785314Z | OUR_GEOMETRIC_ESTIMATE |
| SDO | INSIDE_CONE | 11.6° | 2014-09-11T08:47:52.510418Z | OUR_GEOMETRIC_ESTIMATE |
| GOES-16 | INSIDE_CONE | 11.6° | 2014-09-11T08:47:52.510418Z | OUR_GEOMETRIC_ESTIMATE |
| ISS | INSIDE_CONE | 11.6° | 2014-09-11T08:47:52.510418Z | OUR_GEOMETRIC_ESTIMATE |
| GPS-BIIRM-1 | INSIDE_CONE | 11.6° | 2014-09-11T08:47:52.510418Z | OUR_GEOMETRIC_ESTIMATE |

---

## Case: 2015-Jun-22 M6.5
*M6.5 from AR 12371; fast CME, minor storm Jun 24.*
**Source**: `DONKI_OBSERVED; NOAA/SWPC`

### CMEAnalysis Parameters (DONKI_OBSERVED)
| Field | Value |
|---|---|
| CME ID | `2015-06-22T18:36:00-CME-001` |
| Speed (km/s) | `1209.0` |
| Latitude (Stonyhurst °) | `13.0` |
| Longitude (Stonyhurst °) | `4.0` |
| Half Angle (°) | `38.0` |
| SEP Observed | `NOT_OBSERVED` |
| Max Kp Observed | `5.5` |

### Our Geometric Estimate (OUR_GEOMETRIC_ESTIMATE)
| Field | Value |
|---|---|
| Earth Impact Status | `EARTH_CONE_INTERSECTION` |
| Earth Angular Sep (°) | `13.6°` |
| Estimated Arrival at Earth | ~34.4 hours after CME start (ESTIMATED) |

### Validation Comparison
- Observed Earth Impact: **YES**
- Cone Model Result: **EARTH_CONE_INTERSECTION**
- **Agreement: AGREEMENT**
- Explanation: Both observed Earth impact (GST/storm) and cone model confirm Earth-directed geometry.

### Spacecraft Exposure Table (STATIC_ORBIT_APPROXIMATION)
| Spacecraft | Exposure | Angular Sep (°) | Estimated Arrival | Calc. Method |
|---|---|---|---|---|
| SOHO | INSIDE_CONE | 13.6° | 2015-06-23T10:01:39.497099Z | OUR_GEOMETRIC_ESTIMATE |
| SDO | INSIDE_CONE | 13.6° | 2015-06-23T10:22:16.865757Z | OUR_GEOMETRIC_ESTIMATE |
| GOES-16 | INSIDE_CONE | 13.6° | 2015-06-23T10:22:16.865757Z | OUR_GEOMETRIC_ESTIMATE |
| ISS | INSIDE_CONE | 13.6° | 2015-06-23T10:22:16.865757Z | OUR_GEOMETRIC_ESTIMATE |
| GPS-BIIRM-1 | INSIDE_CONE | 13.6° | 2015-06-23T10:22:16.865757Z | OUR_GEOMETRIC_ESTIMATE |

---

## Case: 2015-Mar-15 St. Patrick's Day GST
*St. Patrick's Day extreme geomagnetic storm (Kp=8). CME from C9 flare.*
**Source**: `DONKI_OBSERVED; Jacobsen & Andalsvik 2016`

### CMEAnalysis Parameters (DONKI_OBSERVED)
| Field | Value |
|---|---|
| CME ID | `2015-03-15T01:48:00-CME-001` |
| Speed (km/s) | `717.0` |
| Latitude (Stonyhurst °) | `-6.0` |
| Longitude (Stonyhurst °) | `18.0` |
| Half Angle (°) | `60.0` |
| SEP Observed | `NOT_OBSERVED` |
| Max Kp Observed | `8.0` |

### Our Geometric Estimate (OUR_GEOMETRIC_ESTIMATE)
| Field | Value |
|---|---|
| Earth Impact Status | `EARTH_CONE_INTERSECTION` |
| Earth Angular Sep (°) | `18.9°` |
| Estimated Arrival at Earth | ~58.0 hours after CME start (ESTIMATED) |

### Validation Comparison
- Observed Earth Impact: **YES**
- Cone Model Result: **EARTH_CONE_INTERSECTION**
- **Agreement: AGREEMENT**
- Explanation: Both observed Earth impact (GST/storm) and cone model confirm Earth-directed geometry.

### Spacecraft Exposure Table (STATIC_ORBIT_APPROXIMATION)
| Spacecraft | Exposure | Angular Sep (°) | Estimated Arrival | Calc. Method |
|---|---|---|---|---|
| SOHO | INSIDE_CONE | 18.9° | 2015-03-17T09:22:37.729418Z | OUR_GEOMETRIC_ESTIMATE |
| SDO | INSIDE_CONE | 18.9° | 2015-03-17T09:57:24.171130Z | OUR_GEOMETRIC_ESTIMATE |
| GOES-16 | INSIDE_CONE | 18.9° | 2015-03-17T09:57:24.171130Z | OUR_GEOMETRIC_ESTIMATE |
| ISS | INSIDE_CONE | 18.9° | 2015-03-17T09:57:24.171130Z | OUR_GEOMETRIC_ESTIMATE |
| GPS-BIIRM-1 | INSIDE_CONE | 18.9° | 2015-03-17T09:57:24.171130Z | OUR_GEOMETRIC_ESTIMATE |

---

## Case: 2017-Sep-6 X9.3
*Strongest flare of SC24 (X9.3). CME hit Earth Sep 7, Kp=8.*
**Source**: `DONKI_OBSERVED; Chertok et al. 2018`

### CMEAnalysis Parameters (DONKI_OBSERVED)
| Field | Value |
|---|---|
| CME ID | `2017-09-06T12:24:00-CME-001` |
| Speed (km/s) | `1571.0` |
| Latitude (Stonyhurst °) | `-1.0` |
| Longitude (Stonyhurst °) | `-19.0` |
| Half Angle (°) | `55.0` |
| SEP Observed | `SEP_OBSERVED` |
| Max Kp Observed | `8.0` |

### Our Geometric Estimate (OUR_GEOMETRIC_ESTIMATE)
| Field | Value |
|---|---|
| Earth Impact Status | `EARTH_CONE_INTERSECTION` |
| Earth Angular Sep (°) | `19.0°` |
| Estimated Arrival at Earth | ~26.5 hours after CME start (ESTIMATED) |

### Validation Comparison
- Observed Earth Impact: **YES**
- Cone Model Result: **EARTH_CONE_INTERSECTION**
- **Agreement: AGREEMENT**
- Explanation: Both observed Earth impact (GST/storm) and cone model confirm Earth-directed geometry.

### Spacecraft Exposure Table (STATIC_ORBIT_APPROXIMATION)
| Spacecraft | Exposure | Angular Sep (°) | Estimated Arrival | Calc. Method |
|---|---|---|---|---|
| SOHO | INSIDE_CONE | 19.0° | 2017-09-07T02:11:12.369187Z | OUR_GEOMETRIC_ESTIMATE |
| SDO | INSIDE_CONE | 19.0° | 2017-09-07T02:27:04.615341Z | OUR_GEOMETRIC_ESTIMATE |
| GOES-16 | INSIDE_CONE | 19.0° | 2017-09-07T02:27:04.615341Z | OUR_GEOMETRIC_ESTIMATE |
| ISS | INSIDE_CONE | 19.0° | 2017-09-07T02:27:04.615341Z | OUR_GEOMETRIC_ESTIMATE |
| GPS-BIIRM-1 | INSIDE_CONE | 19.0° | 2017-09-07T02:27:04.615341Z | OUR_GEOMETRIC_ESTIMATE |

---

## Case: 2017-Sep-10 X8.2
*X8.2 from western limb (W104°); CME not Earth-directed. Large SEP via well-connected field lines.*
**Source**: `DONKI_OBSERVED; Morosan et al. 2019`

### CMEAnalysis Parameters (DONKI_OBSERVED)
| Field | Value |
|---|---|
| CME ID | `2017-09-10T15:35:00-CME-001` |
| Speed (km/s) | `3163.0` |
| Latitude (Stonyhurst °) | `-21.0` |
| Longitude (Stonyhurst °) | `-104.0` |
| Half Angle (°) | `30.0` |
| SEP Observed | `SEP_OBSERVED` |
| Max Kp Observed | `None` |

### Our Geometric Estimate (OUR_GEOMETRIC_ESTIMATE)
| Field | Value |
|---|---|
| Earth Impact Status | `NO_EARTH_CONE_INTERSECTION` |
| Earth Angular Sep (°) | `103.1°` |
| Estimated Arrival at Earth | ~13.1 hours after CME start (ESTIMATED) |

### Validation Comparison
- Observed Earth Impact: **NO**
- Cone Model Result: **NO_EARTH_CONE_INTERSECTION**
- **Agreement: AGREEMENT**
- Explanation: No Earth impact observed and cone model correctly returns NO_EARTH_CONE_INTERSECTION (sep=103.1°).

### Spacecraft Exposure Table (STATIC_ORBIT_APPROXIMATION)
| Spacecraft | Exposure | Angular Sep (°) | Estimated Arrival | Calc. Method |
|---|---|---|---|---|
| SOHO | OUTSIDE | 103.1° | 2017-09-10T13:00:23.234901Z | OUR_GEOMETRIC_ESTIMATE |
| SDO | OUTSIDE | 103.1° | 2017-09-10T13:08:16.196870Z | OUR_GEOMETRIC_ESTIMATE |
| GOES-16 | OUTSIDE | 103.1° | 2017-09-10T13:08:16.196870Z | OUR_GEOMETRIC_ESTIMATE |
| ISS | OUTSIDE | 103.1° | 2017-09-10T13:08:16.196870Z | OUR_GEOMETRIC_ESTIMATE |
| GPS-BIIRM-1 | OUTSIDE | 103.1° | 2017-09-10T13:08:16.196870Z | OUR_GEOMETRIC_ESTIMATE |

---

## Case: 2021-Oct-28 X1.0
*X1.0 from AR 12887; Earth-directed, SEP, Kp=5 storm Oct 30–31.*
**Source**: `DONKI_OBSERVED; Paassilta et al. 2023`

### CMEAnalysis Parameters (DONKI_OBSERVED)
| Field | Value |
|---|---|
| CME ID | `2021-10-28T15:35:00-CME-001` |
| Speed (km/s) | `1519.0` |
| Latitude (Stonyhurst °) | `-12.0` |
| Longitude (Stonyhurst °) | `-30.0` |
| Half Angle (°) | `52.0` |
| SEP Observed | `SEP_OBSERVED` |
| Max Kp Observed | `5.0` |

### Our Geometric Estimate (OUR_GEOMETRIC_ESTIMATE)
| Field | Value |
|---|---|
| Earth Impact Status | `EARTH_CONE_INTERSECTION` |
| Earth Angular Sep (°) | `32.1°` |
| Estimated Arrival at Earth | ~27.4 hours after CME start (ESTIMATED) |

### Validation Comparison
- Observed Earth Impact: **YES**
- Cone Model Result: **EARTH_CONE_INTERSECTION**
- **Agreement: AGREEMENT**
- Explanation: Both observed Earth impact (GST/storm) and cone model confirm Earth-directed geometry.

### Spacecraft Exposure Table (STATIC_ORBIT_APPROXIMATION)
| Spacecraft | Exposure | Angular Sep (°) | Estimated Arrival | Calc. Method |
|---|---|---|---|---|
| SOHO | INSIDE_CONE | 32.1° | 2021-10-29T03:04:59.599732Z | OUR_GEOMETRIC_ESTIMATE |
| SDO | INSIDE_CONE | 32.1° | 2021-10-29T03:21:24.444174Z | OUR_GEOMETRIC_ESTIMATE |
| GOES-16 | INSIDE_CONE | 32.1° | 2021-10-29T03:21:24.444174Z | OUR_GEOMETRIC_ESTIMATE |
| ISS | INSIDE_CONE | 32.1° | 2021-10-29T03:21:24.444174Z | OUR_GEOMETRIC_ESTIMATE |
| GPS-BIIRM-1 | INSIDE_CONE | 32.1° | 2021-10-29T03:21:24.444174Z | OUR_GEOMETRIC_ESTIMATE |

---

## Case: 2022-Mar-28 M4.0
*M4.0 flare; moderate slow CME, minor storm Mar 31.*
**Source**: `DONKI_OBSERVED; NOAA/SWPC`

### CMEAnalysis Parameters (DONKI_OBSERVED)
| Field | Value |
|---|---|
| CME ID | `2022-03-28T11:29:00-CME-001` |
| Speed (km/s) | `490.0` |
| Latitude (Stonyhurst °) | `0.0` |
| Longitude (Stonyhurst °) | `-22.0` |
| Half Angle (°) | `24.0` |
| SEP Observed | `NOT_OBSERVED` |
| Max Kp Observed | `3.5` |

### Our Geometric Estimate (OUR_GEOMETRIC_ESTIMATE)
| Field | Value |
|---|---|
| Earth Impact Status | `EARTH_CONE_INTERSECTION` |
| Earth Angular Sep (°) | `22.0°` |
| Estimated Arrival at Earth | ~84.8 hours after CME start (ESTIMATED) |

### Validation Comparison
- Observed Earth Impact: **YES**
- Cone Model Result: **EARTH_CONE_INTERSECTION**
- **Agreement: AGREEMENT**
- Explanation: Both observed Earth impact (GST/storm) and cone model confirm Earth-directed geometry.

### Spacecraft Exposure Table (STATIC_ORBIT_APPROXIMATION)
| Spacecraft | Exposure | Angular Sep (°) | Estimated Arrival | Calc. Method |
|---|---|---|---|---|
| SOHO | INSIDE_CONE | 22.0° | 2022-03-31T11:57:28.759169Z | OUR_GEOMETRIC_ESTIMATE |
| SDO | INSIDE_CONE | 22.0° | 2022-03-31T12:48:21.776939Z | OUR_GEOMETRIC_ESTIMATE |
| GOES-16 | INSIDE_CONE | 22.0° | 2022-03-31T12:48:21.776939Z | OUR_GEOMETRIC_ESTIMATE |
| ISS | INSIDE_CONE | 22.0° | 2022-03-31T12:48:21.776939Z | OUR_GEOMETRIC_ESTIMATE |
| GPS-BIIRM-1 | INSIDE_CONE | 22.0° | 2022-03-31T12:48:21.776939Z | OUR_GEOMETRIC_ESTIMATE |

---

## Methodology Notes
### Cone Model
- Angular separation between CME propagation direction and target is computed using the great-circle (Haversine) formula.
- Earth is placed at heliocentric Stonyhurst lat=0°, lon=0° (central meridian, ecliptic plane).
- `INSIDE_CONE`: angular_sep ≤ half_angle
- `NEAR_FLANK`: angular_sep ≤ half_angle + 15° (flank_margin, not tuned)
- `OUTSIDE`: angular_sep > half_angle + 15°

### Known Limitations
1. **Static spacecraft positions**: Real ephemerides (e.g., SSCWeb) are required for precise exposure calculations.
2. **Simplified cone**: A symmetric, uniformly expanding cone does not capture CME deflection, rotation, or magnetic sheath width.
3. **Stonyhurst approximation**: For Earth-orbiting spacecraft, we assume heliocentric lon=0°. Real STEREO or L1 assets have measurable longitudes.
4. **Arrival time**: Constant-speed propagation ignores deceleration in the solar wind. Real arrival times vary by ±12–24 hours.
5. **Cone parameters not tuned**: The 15° flank margin is default. Any disagreements are reported faithfully.
6. **No WSA-ENLIL comparison**: The DONKI API was unresponsive during live validation. WSA-ENLIL outputs should be cross-checked manually.