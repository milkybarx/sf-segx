"""Spacecraft Catalog and Geometric CME Exposure calculations."""
import os
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd

class SpacecraftCatalog:
    """Manages the catalog of spacecraft assets and computes CME exposure."""
    
    def __init__(self, csv_path: str = "data/spacecraft/spacecraft.csv"):
        self.csv_path = csv_path
        self.spacecrafts = []
        self._load_catalog()

    def _load_catalog(self):
        """Load the spacecraft asset definitions from CSV."""
        if os.path.exists(self.csv_path):
            try:
                df = pd.read_csv(self.csv_path)
                self.spacecrafts = df.to_dict(orient="records")
            except Exception as e:
                print(f"Error reading spacecraft catalog CSV: {e}")
                self._load_defaults()
        else:
            self._load_defaults()

    def _load_defaults(self):
        """Fallback defaults if CSV is missing."""
        self.spacecrafts = [
            {
                "satellite_id": "SOHO",
                "name": "SOHO",
                "mission": "Solar and Heliospheric Observatory",
                "operator": "ESA/NASA",
                "orbit_type": "L1_HALO",
                "altitude": 1500000.0,
                "inclination": 0.0,
                "longitude": 0.0,
                "eccentricity": 0.0,
                "trajectory_source": "NASA_CCMC"
            },
            {
                "satellite_id": "SDO",
                "name": "SDO",
                "mission": "Solar Dynamics Observatory",
                "operator": "NASA",
                "orbit_type": "GEO",
                "altitude": 35786.0,
                "inclination": 28.5,
                "longitude": -102.0,
                "eccentricity": 0.0001,
                "trajectory_source": "NASA_CCMC"
            },
            {
                "satellite_id": "GOES-16",
                "name": "GOES-16",
                "mission": "Geostationary Operational Environmental Satellite",
                "operator": "NOAA",
                "orbit_type": "GEO",
                "altitude": 35786.0,
                "inclination": 0.0,
                "longitude": -75.2,
                "eccentricity": 0.0,
                "trajectory_source": "NOAA"
            },
            {
                "satellite_id": "GOES-18",
                "name": "GOES-18",
                "mission": "Geostationary Operational Environmental Satellite",
                "operator": "NOAA",
                "orbit_type": "GEO",
                "altitude": 35786.0,
                "inclination": 0.0,
                "longitude": -137.2,
                "eccentricity": 0.0,
                "trajectory_source": "NOAA"
            },
            {
                "satellite_id": "ISS",
                "name": "ISS",
                "mission": "International Space Station",
                "operator": "International",
                "orbit_type": "LEO",
                "altitude": 420.0,
                "inclination": 51.64,
                "longitude": 0.0,
                "eccentricity": 0.0003,
                "trajectory_source": "NASA"
            },
            {
                "satellite_id": "GPS-BIIRM-1",
                "name": "GPS-BIIRM-1",
                "mission": "Global Positioning System",
                "operator": "US Space Force",
                "orbit_type": "MEO",
                "altitude": 20200.0,
                "inclination": 55.0,
                "longitude": -120.0,
                "eccentricity": 0.005,
                "trajectory_source": "USSF"
            }
        ]

    def get_spacecraft_position(self, sat_id: str, timestamp: str = None) -> Dict[str, Any]:
        """Get the heliocentric Stonyhurst position of a spacecraft in degrees and distance (AU).
        
        Priority:
        A. NASA SSCWeb ephemeris (where supported - not fully implemented, mocked)
        B. Another validated real ephemeris source
        C. Existing static catalog approximation
        D. UNKNOWN
        """
        sat_id = sat_id.strip().upper()
        # Default fallback
        pos = {"latitude": 0.0, "longitude": 0.0, "distance_au": 1.0, "position_source": "STATIC_ORBIT_APPROXIMATION"}
        
        if sat_id == "PARKER_SP":
            pos = {"latitude": 0.0, "longitude": -45.0, "distance_au": 0.25, "position_source": "STATIC_ORBIT_APPROXIMATION"}
        elif sat_id == "SOLAR_ORB":
            pos = {"latitude": 15.0, "longitude": 30.0, "distance_au": 0.7, "position_source": "STATIC_ORBIT_APPROXIMATION"}
        elif sat_id == "SOHO":
            pos = {"latitude": 0.0, "longitude": 0.0, "distance_au": 0.99, "position_source": "STATIC_ORBIT_APPROXIMATION"}
            
        return pos
    def calculate_cme_exposure(self, sat_id: str, cme_lat: float, cme_lon: float,
                              cme_half_angle: float, cme_speed: float,
                              cme_start_time: str, nasa_model_impact: dict = None) -> Dict[str, Any]:
        """Evaluate exposure category (INSIDE_CONE, NEAR_FLANK, OUTSIDE) and estimated arrival.
        """
        from .cme_geometry import CMEGeometryModel
        geom = CMEGeometryModel()
        
        pos = self.get_spacecraft_position(sat_id, cme_start_time)
        sc_lat = pos["latitude"]
        sc_lon = pos["longitude"]
        sc_dist_au = pos["distance_au"]
        pos_source = pos.get("position_source", "UNKNOWN")
        
        exposure_type, angular_sep = geom.evaluate_exposure(cme_lat, cme_lon, cme_half_angle, sc_lat, sc_lon)
        
        arrival_time_str = "N/A"
        travel_time_hours = float("nan")
        calc_method = "UNKNOWN"
        
        if nasa_model_impact and nasa_model_impact.get("spacecraft_id") == sat_id:
            arrival_time_str = nasa_model_impact.get("arrival_time", "N/A")
            calc_method = "NASA_MODEL"
            exposure_type = "INSIDE_CONE"  # If NASA predicted impact, we are exposed
        elif cme_speed > 0:
            hrs, mth = geom.estimate_arrival_time(cme_speed, sc_dist_au)
            travel_time_hours = hrs if hrs is not None else float("nan")
            calc_method = mth
            try:
                clean_time = cme_start_time.replace('Z', '')
                if 'T' in clean_time:
                    dt = datetime.fromisoformat(clean_time)
                else:
                    dt = datetime.strptime(clean_time, "%Y-%m-%d %H:%M")
                
                if hrs:
                    arrival_dt = dt + timedelta(hours=hrs)
                    arrival_time_str = arrival_dt.isoformat() + "Z"
            except Exception:
                pass
                
        return {
            "satellite_id": sat_id,
            "exposure_type": exposure_type,
            "angular_separation": float(angular_sep),
            "travel_time_hours": travel_time_hours,
            "estimated_arrival": arrival_time_str,
            "calculation_method": calc_method,
            "trajectory_source": pos_source,
            "event_source": "DONKI_OBSERVED" if cme_start_time else "UNKNOWN"
        }

    def evaluate_all_spacecraft(self, cme_lat: float, cme_lon: float,
                                cme_half_angle: float, cme_speed: float,
                                cme_start_time: str) -> List[Dict[str, Any]]:
        """Evaluate CME exposure and arrival statistics for all catalogued spacecraft assets."""
        results = []
        for sc in self.spacecrafts:
            sat_id = sc["satellite_id"]
            exposure_data = self.calculate_cme_exposure(sat_id, cme_lat, cme_lon, cme_half_angle, cme_speed, cme_start_time)
            results.append({**sc, **exposure_data})
        return results
