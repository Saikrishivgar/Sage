"""
Navigation Constraints Module
==============================
Basic motion constraints for vehicle navigation:
- Zero Velocity Update (ZUPT)
- Maximum speed limit
- Non-holonomic constraint interface (Phase 2)
"""

import numpy as np
from typing import Optional


def zero_velocity_update(speed: float, acc_mag: float,
                         gyro_mag: float,
                         acc_threshold: float = 0.5,
                         gyro_threshold: float = 0.05) -> bool:
    """
    Detect if vehicle is stationary (ZUPT).
    
    Parameters
    ----------
    speed : float
        Current estimated speed (m/s).
    acc_mag : float
        Acceleration magnitude deviation from gravity.
    gyro_mag : float
        Gyroscope magnitude (rad/s).
    
    Returns
    -------
    is_stationary : bool
    """
    acc_deviation = abs(acc_mag - 9.81)
    return (acc_deviation < acc_threshold and 
            gyro_mag < gyro_threshold and 
            speed < 1.0)


def apply_speed_constraint(velocity: np.ndarray,
                           max_speed: float = 50.0) -> np.ndarray:
    """
    Clamp velocity to maximum speed.
    
    Parameters
    ----------
    velocity : np.ndarray, shape (2,)
        [vx, vy] in m/s.
    max_speed : float
        Maximum allowed speed (m/s). Default 50 m/s (~180 km/h).
    
    Returns
    -------
    constrained_velocity : np.ndarray
    """
    speed = np.linalg.norm(velocity)
    if speed > max_speed:
        return velocity * (max_speed / speed)
    return velocity
