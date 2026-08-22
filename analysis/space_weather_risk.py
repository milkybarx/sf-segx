"""Space Weather and Spacecraft Subsystem Risk Analyzer."""
from typing import Dict, Any, List, Optional
import numpy as np

class SpaceWeatherRiskAnalyzer:
    """Computes space weather component risks and spacecraft subsystem risks."""
    
    def __init__(self):
        pass

    def evaluate_component_risks(self, flare: Optional[Dict[str, Any]],
                                 cme_analysis: Optional[Dict[str, Any]],
                                 sep: Optional[Dict[str, Any]],
                                 gst: Optional[Dict[str, Any]],
                                 rbe: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute scores (0-100) and bands (LOW, MODERATE, HIGH, EXTREME) for space-weather domains."""
        
        # 1. Ionospheric / HF Radio Risk (Depends primarily on flare class)
        ionosphere_score = 10.0
        if flare:
            fl_class = str(flare.get("class_type", "B")).upper()
            if fl_class.startswith("X"):
                try:
                    mag = float(fl_class[1:])
                    ionosphere_score = min(80.0 + mag * 2.0, 100.0)
                except ValueError:
                    ionosphere_score = 90.0
            elif fl_class.startswith("M"):
                try:
                    mag = float(fl_class[1:])
                    ionosphere_score = min(50.0 + mag * 3.0, 79.0)
                except ValueError:
                    ionosphere_score = 65.0
            elif fl_class.startswith("C"):
                ionosphere_score = 35.0
            else:
                ionosphere_score = 20.0
                
        # 2. Magnetospheric Risk (CME speed + Earth-directed evidence + GST + IPS)
        magnetosphere_score = 10.0
        if gst:
            # Strong geomagnetic storm detected
            kp = float(gst.get("kp_max", 0.0))
            magnetosphere_score = max(magnetosphere_score, min(kp * 11.0, 100.0))
            
        if cme_analysis:
            speed = float(cme_analysis.get("speed", 0.0))
            is_most_accurate = cme_analysis.get("is_most_accurate", False)
            
            # Speed contribution: 1000 km/s -> score 50, 2000 km/s -> score 90
            speed_contrib = min(speed / 22.0, 95.0)
            
            # Check direction: if longitude is large, CME is not Earth-directed (glancing or limb CME)
            cme_lon = abs(float(cme_analysis.get("longitude", 0.0) or 0.0))
            cme_lat = abs(float(cme_analysis.get("latitude", 0.0) or 0.0))
            
            if cme_lon > 45.0 or cme_lat > 35.0:
                # Glancing or limb CME - reduce geomagnetic storm risk
                speed_contrib *= 0.3
                
            magnetosphere_score = max(magnetosphere_score, speed_contrib)
            
        # 3. GNSS Risk (Combines flare radiation risk + geomagnetic storm risk)
        # GNSS suffers from both solar radio bursts (solar flare) and ionospheric scintillation (geomagnetic storm)
        gnss_score = 0.4 * ionosphere_score + 0.6 * magnetosphere_score
        
        # 4. Radiation Risk (SEP + RBE linked records)
        radiation_score = 10.0
        radiation_confidence = 0.5
        if sep:
            radiation_score = max(radiation_score, 75.0)
            radiation_confidence = 0.85
        if rbe:
            radiation_score = max(radiation_score, 60.0)
            radiation_confidence = 0.80
        if sep and rbe:
            radiation_score = 90.0
            
        # 5. Thermospheric Drag Risk (Depends on geomagnetic storm heating)
        drag_score = 10.0
        if gst:
            kp = float(gst.get("kp_max", 0.0))
            drag_score = max(drag_score, min(kp * 10.0, 100.0))
        elif magnetosphere_score > 30.0:
            drag_score = magnetosphere_score * 0.8
            
        # Helper to convert score to band
        def get_band(score: float) -> str:
            if score >= 80.0: return "EXTREME"
            elif score >= 55.0: return "HIGH"
            elif score >= 30.0: return "MODERATE"
            else: return "LOW"
            
        # Overall Risk is the maximum of the component risks
        scores = {
            "ionosphere": ionosphere_score,
            "magnetosphere": magnetosphere_score,
            "gnss": gnss_score,
            "radiation": radiation_score,
            "drag": drag_score
        }
        
        overall_score = max(scores.values())
        overall_band = get_band(overall_score)
        
        # Determine primary and secondary drivers
        sorted_drivers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary = sorted_drivers[0][0].upper()
        secondary = sorted_drivers[1][0].upper() if len(sorted_drivers) > 1 else "NONE"
        
        # Override primary driver with specific physical reason
        primary_reason = f"Elevated {primary} activity"
        if primary == "IONOSPHERE" and flare:
            primary_reason = f"Strong Solar Flare ({flare.get('class_type')})"
        elif primary == "MAGNETOSPHERE" and cme_analysis:
            primary_reason = f"Fast CME ({cme_analysis.get('speed')} km/s)"
        elif primary == "MAGNETOSPHERE" and gst:
            primary_reason = f"Geomagnetic Storm (Kp max {gst.get('kp_max')})"
        elif primary == "RADIATION" and sep:
            primary_reason = "Solar Energetic Particle (SEP) Event"
            
        return {
            "component_scores": scores,
            "component_bands": {k: get_band(v) for k, v in scores.items()},
            "overall_score": float(overall_score),
            "overall_band": overall_band,
            "primary_driver": primary_reason,
            "secondary_driver": f"Elevated {secondary} risk",
            "radiation_confidence": radiation_confidence
        }

    def evaluate_spacecraft_vulnerability(self, spacecraft: Dict[str, Any],
                                          comp_risks: Dict[str, Any]) -> Dict[str, Any]:
        """Compute subsystem risks for a single spacecraft based on its orbit and exposure type.
        
        Formula: Subsystem Risk = Hazard * Exposure_Factor * Subsystem_Vulnerability
        """
        exposure = spacecraft.get("exposure_type", "OUTSIDE")
        orbit = spacecraft.get("orbit_type", "LEO").upper()
        
        # Exposure Factor mapping
        exp_factor = 0.0
        if exposure == "INSIDE_CONE":
            exp_factor = 1.0
        elif exposure == "NEAR_FLANK":
            exp_factor = 0.5
        elif exposure == "OUTSIDE":
            exp_factor = 0.1  # background risk
            
        # Get component hazard scores (0-100)
        c_scores = comp_risks["component_scores"]
        h_rad = c_scores["radiation"]
        h_mag = c_scores["magnetosphere"]
        h_ion = c_scores["ionosphere"]
        h_gnss = c_scores["gnss"]
        h_drag = c_scores["drag"]
        
        # Orbit Vulnerability Coefficients
        # LEO: ISS is inside magnetosphere (low radiation unless extreme, high drag)
        # GEO/MEO: SDO, GOES, GPS are outside main atmosphere (zero drag, high radiation, moderate mag)
        # L1: SOHO is completely unprotected by Earth's field (extreme radiation/attitude risk, zero drag)
        v_rad = 0.8
        v_comm = 0.6
        v_gnss = 0.6
        v_att = 0.5
        v_pwr = 0.5
        v_sens = 0.5
        v_drag = 0.0
        
        if orbit == "LEO":
            v_rad = 0.3      # Protected by geomagnetic field
            v_comm = 0.7     # Heavy scintillation through F-region
            v_gnss = 0.8     # Critical for orbit tracking
            v_att = 0.4      # Magnetic torquers vulnerable to field changes
            v_pwr = 0.3
            v_sens = 0.3
            v_drag = 1.0     # Dense atmosphere heating expands thermosphere
        elif orbit == "GEO":
            v_rad = 0.8      # Exposed to outer radiation belt
            v_comm = 0.8     # Long satellite link paths
            v_gnss = 0.4
            v_att = 0.6      # Star tracker noise
            v_pwr = 0.7      # Rapid solar panel aging
            v_sens = 0.8
            v_drag = 0.0     # Negligible atmosphere
        elif orbit == "MEO":
            v_rad = 0.9      # Heart of radiation belts (GPS)
            v_comm = 0.6
            v_gnss = 0.9     # GPS satellites themselves!
            v_att = 0.5
            v_pwr = 0.8
            v_sens = 0.7
            v_drag = 0.0
        elif orbit in ["L1_HALO", "HELIOCENTRIC"]:
            v_rad = 1.0      # Zero magnetospheric shielding
            v_comm = 0.9     # Critical deep space links
            v_gnss = 0.1     # Doesn't use GPS
            v_att = 0.8      # Solar wind torque, high sensor noise
            v_pwr = 0.9      # Severe degradation from proton flares
            v_sens = 1.0     # Highly sensitive instruments
            v_drag = 0.0
            
        # Compute Subsystem Scores (0-100)
        # Combine Hazard, Exposure Factor, and Vulnerability
        r_rad = h_rad * exp_factor * v_rad
        r_comm = h_ion * exp_factor * v_comm
        r_gnss = h_gnss * exp_factor * v_gnss
        r_att = h_mag * exp_factor * v_att
        # Power degradation is primarily driven by SEP (Radiation hazard)
        r_pwr = h_rad * exp_factor * v_pwr
        r_sens = max(h_rad, h_ion) * exp_factor * v_sens
        r_drag = h_drag * exp_factor * v_drag
        
        subsystems = {
            "radiation": r_rad,
            "communications": r_comm,
            "gnss": r_gnss,
            "attitude": r_att,
            "power": r_pwr,
            "sensor": r_sens,
            "drag": r_drag
        }
        
        # Determine overall spacecraft risk
        overall_score = max(subsystems.values())
        
        def get_band(score: float) -> str:
            if score >= 75.0: return "EXTREME"
            elif score >= 50.0: return "HIGH"
            elif score >= 25.0: return "MODERATE"
            else: return "LOW"
            
        return {
            "satellite_id": spacecraft.get("satellite_id"),
            "name": spacecraft.get("name"),
            "subsystem_scores": {k: float(v) for k, v in subsystems.items()},
            "subsystem_bands": {k: get_band(v) for k, v in subsystems.items()},
            "overall_spacecraft_score": float(overall_score),
            "overall_spacecraft_risk": get_band(overall_score),
            "vulnerable_subsystem": max(subsystems, key=subsystems.get).upper()
        }
