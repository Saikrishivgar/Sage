"""
Coordinate Transform Module
=============================
Conversions between:
- Lat/Lon (WGS84) and local tangent plane (ENU meters)
- Body frame and navigation frame
"""

import numpy as np
from typing import Tuple


# WGS84 constants
WGS84_A = 6378137.0  # Semi-major axis (m)
WGS84_F = 1 / 298.257223563  # Flattening


def latlon_to_local(lat: np.ndarray, lon: np.ndarray,
                    lat0: float, lon0: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert latitude/longitude to local tangent plane coordinates (meters).
    
    Uses flat-earth approximation (valid for distances < ~10 km).
    
    Parameters
    ----------
    lat, lon : np.ndarray
        Latitude and longitude in degrees.
    lat0, lon0 : float
        Origin latitude and longitude in degrees.
    
    Returns
    -------
    x : np.ndarray
        East displacement in meters.
    y : np.ndarray
        North displacement in meters.
    """
    m_per_deg_lat = 111132.92
    m_per_deg_lon = 111132.92 * np.cos(np.radians(lat0))
    
    x = (lon - lon0) * m_per_deg_lon  # East
    y = (lat - lat0) * m_per_deg_lat  # North
    
    return x, y


def local_to_latlon(x: np.ndarray, y: np.ndarray,
                    lat0: float, lon0: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert local tangent plane coordinates back to lat/lon.
    
    Parameters
    ----------
    x : np.ndarray
        East displacement in meters.
    y : np.ndarray
        North displacement in meters.
    lat0, lon0 : float
        Origin latitude and longitude in degrees.
    
    Returns
    -------
    lat, lon : np.ndarray
        Latitude and longitude in degrees.
    """
    m_per_deg_lat = 111132.92
    m_per_deg_lon = 111132.92 * np.cos(np.radians(lat0))
    
    lat = lat0 + y / m_per_deg_lat
    lon = lon0 + x / m_per_deg_lon
    
    return lat, lon


def haversine_distance(lat1: float, lon1: float,
                       lat2: float, lon2: float) -> float:
    """
    Calculate great-circle distance between two points using Haversine formula.
    
    Parameters
    ----------
    lat1, lon1, lat2, lon2 : float
        Coordinates in degrees.
    
    Returns
    -------
    distance : float
        Distance in meters.
    """
    R = 6371000  # Earth radius in meters
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def compute_heading(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """
    Compute heading from displacement components.
    
    Convention: 0 = North, pi/2 = East (navigation convention).
    
    Parameters
    ----------
    dx : np.ndarray
        East displacement.
    dy : np.ndarray
        North displacement.
    
    Returns
    -------
    heading : np.ndarray
        Heading in radians [0, 2*pi).
    """
    heading = np.arctan2(dx, dy)  # atan2(East, North) for navigation convention
    heading = heading % (2 * np.pi)
    return heading
