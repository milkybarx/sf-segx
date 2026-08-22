"""CME Geometry and Earth/Spacecraft Exposure."""
import math
import numpy as np
from typing import Dict, Any, Tuple, Optional


def angular_difference(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle angular distance between two points in degrees."""
    # Convert to radians
    phi1, lambda1 = math.radians(lat1), math.radians(lon1)
    phi2, lambda2 = math.radians(lat2), math.radians(lon2)
    
    # Haversine formula
    dphi = phi2 - phi1
    dlambda = lambda2 - lambda1
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return math.degrees(c)

class CMEGeometryModel:
    """Simplified geometric cone model for CME propagation and exposure."""
    
    def __init__(self, flank_margin: float = 15.0):
        self.flank_margin = flank_margin
        
    def evaluate_exposure(self, 
                          cme_lat: float, 
                          cme_lon: float, 
                          cme_half_angle: float, 
                          target_lat: float, 
                          target_lon: float) -> Tuple[str, float]:
        """
        Evaluate if a target (Spacecraft/Earth) is inside the CME cone.
        Returns (exposure_class, angular_separation).
        """
        if any(x is None for x in [cme_lat, cme_lon, cme_half_angle, target_lat, target_lon]):
            return "UNKNOWN", float('nan')
            
        sep = angular_difference(cme_lat, cme_lon, target_lat, target_lon)
        
        if sep <= cme_half_angle:
            return "INSIDE_CONE", sep
        elif sep <= cme_half_angle + self.flank_margin:
            return "NEAR_FLANK", sep
        else:
            return "OUTSIDE", sep

    def estimate_arrival_time(self, cme_speed_km_s: float, distance_au: float = 1.0) -> Tuple[Optional[float], str]:
        """
        Estimate arrival time (in hours) based on constant speed.
        1 AU = 149,597,870.7 km
        Returns (hours, calculation_method)
        """
        if not cme_speed_km_s or cme_speed_km_s <= 0:
            return None, "UNKNOWN"
            
        distance_km = distance_au * 149597870.7
        seconds = distance_km / cme_speed_km_s
        hours = seconds / 3600.0
        return hours, "OUR_GEOMETRIC_ESTIMATE"

    def evaluate_earth_impact(self, 
                              cme_lat: float, 
                              cme_lon: float, 
                              cme_half_angle: float, 
                              nasa_model_impact: bool = False) -> str:
        """
        Determine Earth impact status.
        NASA Model (WSA-ENLIL) takes precedence.
        """
        if nasa_model_impact:
            return "EARTH_IMPACT_NASA_MODEL"
            
        # Earth is always at heliocentric lat 0, lon 0 (or close enough for this simple model relative to solar central meridian)
        # Assuming coordinates are relative to Earth's view (Stonyhurst)
        exposure, _ = self.evaluate_exposure(cme_lat, cme_lon, cme_half_angle, 0.0, 0.0)
        
        if exposure in ["INSIDE_CONE", "NEAR_FLANK"]:
            return "EARTH_CONE_INTERSECTION"
        elif exposure == "OUTSIDE":
            return "NO_EARTH_CONE_INTERSECTION"
        else:
            return "UNKNOWN"
