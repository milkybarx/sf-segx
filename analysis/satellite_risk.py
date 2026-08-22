"""Spacecraft Subsystem Risk Analyzer."""
from typing import Dict, Any

class SatelliteRiskAnalyzer:
    """Computes specific subsystem risks for a spacecraft based on its exposure to space weather events."""
    
    def evaluate_subsystem_risks(self,
                                 exposure_data: Dict[str, Any],
                                 env_risks: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate subsystem risks.
        exposure_data: From calculate_cme_exposure (includes exposure_type, trajectory_source, etc.)
        env_risks: From space_weather_risk (ionosphere, magnetosphere, gnss, radiation, drag)
        """
        exposure_type = exposure_data.get("exposure_type", "UNKNOWN")
        
        # Exposure multiplier
        exp_mult = 0.1
        if exposure_type == "INSIDE_CONE":
            exp_mult = 1.0
        elif exposure_type == "NEAR_FLANK":
            exp_mult = 0.5
            
        # Environmental scores
        rad_env = env_risks.get("component_scores", {}).get("radiation", 10.0)
        mag_env = env_risks.get("component_scores", {}).get("magnetosphere", 10.0)
        ion_env = env_risks.get("component_scores", {}).get("ionosphere", 10.0)
        drag_env = env_risks.get("component_scores", {}).get("drag", 10.0)
        
        # Subsystem risks (0-100)
        # Communications: primarily affected by ionosphere (flare) and SEP (radiation)
        comms_risk = min(100.0, (ion_env * 0.7 + rad_env * 0.3 * exp_mult))
        
        # GNSS Receiver: affected by ionosphere and magnetosphere
        gnss_rx_risk = min(100.0, (ion_env * 0.5 + mag_env * 0.5 * exp_mult))
        
        # Attitude Control: surface charging (magnetosphere) and single event upsets (radiation)
        attitude_risk = min(100.0, (mag_env * 0.5 + rad_env * 0.5) * exp_mult)
        
        # Power System: deep dielectric charging (radiation) and solar array degradation
        power_risk = min(100.0, rad_env * exp_mult)
        
        # Sensor Risk: radiation background noise
        sensor_risk = min(100.0, rad_env * 0.8 * exp_mult)
        
        # Drag Risk: thermospheric heating (magnetosphere) - mainly LEO
        drag_risk = min(100.0, drag_env * exp_mult)
        
        scores = {
            "communications_risk": comms_risk,
            "gnss_receiver_risk": gnss_rx_risk,
            "attitude_control_risk": attitude_risk,
            "power_system_risk": power_risk,
            "sensor_risk": sensor_risk,
            "drag_risk": drag_risk
        }
        
        max_risk = max(scores.values())
        most_vuln = max(scores, key=scores.get) if max_risk > 20.0 else "None"
        
        def get_level(s):
            if s >= 80: return "EXTREME"
            if s >= 55: return "HIGH"
            if s >= 30: return "MODERATE"
            return "LOW"
            
        return {
            "subsystem_scores": scores,
            "overall_risk_score": max_risk,
            "overall_risk_level": get_level(max_risk),
            "most_vulnerable_subsystem": most_vuln,
            "primary_threat": "Radiation" if rad_env > mag_env else "Geomagnetic",
            "provenance": {
                "event_source": exposure_data.get("event_source", "UNKNOWN"),
                "trajectory_source": exposure_data.get("trajectory_source", "UNKNOWN"),
                "calculation_method": exposure_data.get("calculation_method", "UNKNOWN")
            }
        }
