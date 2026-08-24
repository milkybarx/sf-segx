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
        """Get heliocentric Stonyhurst position of a spacecraft.
        
        For L1/halo-orbit missions we use well-known positions.
        For Earth-orbiting assets (GEO/MEO/LEO) we place them at Earth's position
        (lat≈0, lon≈0 in Stonyhurst) plus their *sub-satellite longitude offset*
        so they spread around Earth rather than all stacking at (0,0).
        """
        sat_id_upper = sat_id.strip().upper()

        # --- Known interplanetary missions with real positions ---
        KNOWN = {
            "PARKER_SP":  {"latitude": 0.0,  "longitude": -45.0, "distance_au": 0.25},
            "PARKER SOLAR PROBE": {"latitude": 0.0, "longitude": -45.0, "distance_au": 0.25},
            "SOLAR_ORB":  {"latitude": 15.0, "longitude":  30.0, "distance_au": 0.70},
            "SOLAR ORBITER": {"latitude": 15.0, "longitude": 30.0, "distance_au": 0.70},
            "SOHO":       {"latitude": 0.0,  "longitude":   0.0, "distance_au": 0.99},
            "STEREO-A":   {"latitude": 0.0,  "longitude": -100.0,"distance_au": 1.00},
            "STEREO-B":   {"latitude": 0.0,  "longitude":  130.0,"distance_au": 1.00},
        }
        if sat_id_upper in KNOWN:
            pos = KNOWN[sat_id_upper].copy()
            pos["position_source"] = "STATIC_ORBIT_APPROXIMATION"
            return pos

        # --- Earth-orbiting satellites: find their entry in catalog for longitude ---
        sc_entry = next((s for s in self.spacecrafts
                         if s["satellite_id"].strip().upper() == sat_id_upper), None)
        if sc_entry:
            orbit_type = sc_entry.get("orbit_type", "GEO")
            sat_lon    = float(sc_entry.get("longitude", 0.0))
            # In Stonyhurst coords Earth is at lon=0. A GEO satellite at -75° Earth-longitude
            # is ~0° in heliocentric coords (it's glued to Earth), but we spread them slightly
            # so the cone model can distinguish them. We map satellite geographic lon → a small
            # heliocentric offset (±30° max) so they don't all land at exactly (0,0).
            hc_lon_offset = (sat_lon / 180.0) * 30.0   # maps ±180° geo → ±30° helio
            dist = {"GEO": 1.00, "MEO": 1.00, "LEO": 1.00, "L1_HALO": 0.99}.get(orbit_type, 1.00)
            return {
                "latitude": 0.0,
                "longitude": round(hc_lon_offset, 2),
                "distance_au": dist,
                "position_source": "STATIC_ORBIT_APPROXIMATION",
            }

        # fallback
        return {"latitude": 0.0, "longitude": 0.0, "distance_au": 1.0,
                "position_source": "STATIC_ORBIT_APPROXIMATION"}
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
