"""Heliographic coordinate conversions for full-disk solar images."""
import re
import numpy as np
from typing import Tuple, Optional


def parse_stonyhurst_string(coord_str: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    """Parse a Stonyhurst coordinate string (e.g., 'N18W45', 'S20E10', 'N03E22') into (lat, lon) in degrees.
    
    Returns (None, None) if parsing fails.
    - North is positive latitude, South is negative latitude.
    - West is positive longitude, East is negative longitude.
    """
    if not coord_str or not isinstance(coord_str, str):
        return None, None
        
    coord_str = coord_str.strip().upper()
    
    # Matches patterns like N18W45, S20E10, N15, E20
    # Let's write a flexible regex
    pattern = r'(?P<ns>[NS])?(?P<ns_val>\d+)?(?P<ew>[EW])?(?P<ew_val>\d+)?'
    match = re.match(pattern, coord_str)
    
    if not match or not coord_str:
        return None, None
        
    lat = None
    lon = None
    
    # Try parsing latitude
    ns = match.group('ns')
    ns_val = match.group('ns_val')
    if ns and ns_val:
        val = float(ns_val)
        lat = val if ns == 'N' else -val
        
    # Try parsing longitude
    ew = match.group('ew')
    ew_val = match.group('ew_val')
    if ew and ew_val:
        val = float(ew_val)
        # West is positive, East is negative in standard Stonyhurst longitude representation
        lon = val if ew == 'W' else -val
        
    return lat, lon


def pixel_to_stonyhurst(x: float, y: float, cx: float, cy: float, radius: float) -> Tuple[float, float]:
    """Convert pixel (x, y) coordinates to Stonyhurst latitude and longitude in degrees.
    
    Assumes North is up (y=0 at top) and West is to the right (x=max at right).
    Returns (NaN, NaN) if coordinates are outside the solar disk radius.
    """
    if radius <= 0:
        return float('nan'), float('nan')
        
    # Translate relative to disk center, normalize by disk radius
    # Invert Y because pixel coordinates grow downwards
    x_rel = (x - cx) / radius
    y_rel = (cy - y) / radius
    
    r = np.sqrt(x_rel**2 + y_rel**2)
    if r > 1.0:
        # Off disk limb
        return float('nan'), float('nan')
        
    # Latitude (theta)
    # y_rel = sin(theta)
    lat_rad = np.arcsin(np.clip(y_rel, -1.0, 1.0))
    lat_deg = float(np.degrees(lat_rad))
    
    # Longitude (phi)
    # x_rel = cos(theta) * sin(phi)
    cos_lat = np.cos(lat_rad)
    if np.abs(cos_lat) < 1e-6:
        # At the exact poles, longitude is singular
        lon_deg = 0.0
    else:
        sin_lon = x_rel / cos_lat
        lon_rad = np.arcsin(np.clip(sin_lon, -1.0, 1.0))
        lon_deg = float(np.degrees(lon_rad))
        
    return lat_deg, lon_deg


def stonyhurst_to_pixel(lat: float, lon: float, cx: float, cy: float, radius: float) -> Tuple[float, float]:
    """Convert Stonyhurst latitude and longitude in degrees to pixel (x, y) coordinates."""
    if radius <= 0 or np.isnan(lat) or np.isnan(lon):
        return float('nan'), float('nan')
        
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    # Spherical to orthographic coordinates
    x_rel = np.cos(lat_rad) * np.sin(lon_rad)
    y_rel = np.sin(lat_rad)
    
    x = cx + x_rel * radius
    y = cy - y_rel * radius
    
    return float(x), float(y)
