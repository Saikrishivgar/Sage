"""
Sensor Calibration Module
==========================
Static bias estimation, scale factor correction.
Interface for future multi-position calibration.
"""

import numpy as np
from typing import Dict, Tuple


def estimate_static_bias(data: np.ndarray, n_samples: int = 100) -> float:
    """
    Estimate static bias from initial stationary period.
    
    Parameters
    ----------
    data : np.ndarray
        Sensor data array.
    n_samples : int
        Number of initial samples assumed stationary.
    
    Returns
    -------
    bias : float
        Estimated static bias.
    """
    return float(np.mean(data[:n_samples]))


def estimate_noise_std(data: np.ndarray, n_samples: int = 100) -> float:
    """Estimate sensor noise standard deviation from stationary data."""
    return float(np.std(data[:n_samples]))


def full_calibration(acc_data: np.ndarray, gyro_data: np.ndarray,
                     n_static: int = 100) -> Dict:
    """
    Perform full sensor calibration.
    
    Parameters
    ----------
    acc_data : np.ndarray, shape (N, 3)
        Accelerometer [ax, ay, az].
    gyro_data : np.ndarray, shape (N, 3)
        Gyroscope [gx, gy, gz].
    n_static : int
        Number of initial stationary samples.
    
    Returns
    -------
    cal : dict
        Calibration parameters.
    """
    cal = {
        'acc_bias': np.mean(acc_data[:n_static], axis=0),
        'acc_noise_std': np.std(acc_data[:n_static], axis=0),
        'gyro_bias': np.mean(gyro_data[:n_static], axis=0),
        'gyro_noise_std': np.std(gyro_data[:n_static], axis=0),
        'gravity_magnitude': float(np.mean(np.linalg.norm(acc_data[:n_static], axis=1))),
        'n_static_samples': n_static,
    }
    
    return cal


def apply_calibration(data: np.ndarray, bias: np.ndarray,
                      scale: np.ndarray = None) -> np.ndarray:
    """
    Apply calibration: remove bias and optionally apply scale factor.
    
    corrected = (raw - bias) * scale
    """
    corrected = data - bias
    if scale is not None:
        corrected = corrected * scale
    return corrected
