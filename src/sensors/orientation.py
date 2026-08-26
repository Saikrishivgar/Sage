"""
Orientation Estimation Module
==============================
Simplified phone-to-vehicle frame transformation.

Current implementation: assumes phone is flat on dashboard.
Future: full quaternion-based orientation from accelerometer + gyroscope.

Vehicle frame convention:
    X = forward (direction of travel)
    Y = lateral (left positive)
    Z = vertical (up positive)
"""

import numpy as np
from typing import Tuple


def get_rotation_matrix_flat() -> np.ndarray:
    """
    Rotation matrix for phone lying flat on dashboard.
    
    Phone frame (Android convention):
        x = right
        y = forward (top of phone)
        z = up (screen)
    
    Vehicle frame:
        X = forward
        Y = left
        Z = up
    
    Mapping:
        Vehicle_X = Phone_y (forward)
        Vehicle_Y = -Phone_x (left)
        Vehicle_Z = Phone_z (up)
    """
    R = np.array([
        [0,  1,  0],   # Vehicle X = Phone Y
        [-1, 0,  0],   # Vehicle Y = -Phone X
        [0,  0,  1],   # Vehicle Z = Phone Z
    ], dtype=float)
    return R


def phone_to_vehicle(acc_phone: np.ndarray, orientation: str = 'flat') -> np.ndarray:
    """
    Transform accelerometer data from phone frame to vehicle frame.
    
    Parameters
    ----------
    acc_phone : np.ndarray, shape (N, 3)
        Accelerometer data in phone frame [ax, ay, az].
    orientation : str
        Phone orientation: 'flat', 'portrait', 'landscape'.
        Currently only 'flat' is fully implemented.
    
    Returns
    -------
    acc_vehicle : np.ndarray, shape (N, 3)
        Accelerometer in vehicle frame [forward, lateral, vertical].
    """
    if orientation == 'flat':
        R = get_rotation_matrix_flat()
    else:
        # Phase 2: implement other orientations
        # For now, use identity (assume aligned)
        R = np.eye(3)
    
    # Apply rotation: acc_vehicle = R @ acc_phone.T
    acc_vehicle = (R @ acc_phone.T).T
    return acc_vehicle


def remove_gravity(acc_vehicle: np.ndarray, gravity: float = 9.81) -> np.ndarray:
    """
    Remove gravity component from vertical axis.
    
    Assumes vehicle frame with Z = up.
    
    Parameters
    ----------
    acc_vehicle : np.ndarray, shape (N, 3)
        Acceleration in vehicle frame [forward, lateral, vertical].
    gravity : float
        Gravity magnitude (m/s²).
    
    Returns
    -------
    acc_linear : np.ndarray, shape (N, 3)
        Linear acceleration with gravity removed.
    """
    acc_linear = acc_vehicle.copy()
    acc_linear[:, 2] -= gravity
    return acc_linear


def estimate_tilt(acc_static: np.ndarray) -> Tuple[float, float]:
    """
    Estimate phone tilt from static accelerometer reading.
    
    Returns roll and pitch angles in radians.
    """
    ax, ay, az = np.mean(acc_static, axis=0)
    
    # Roll (rotation around forward axis)
    roll = np.arctan2(ay, az)
    
    # Pitch (rotation around lateral axis)
    pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
    
    return float(roll), float(pitch)
