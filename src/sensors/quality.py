"""
Sensor Quality Assessment Module
==================================
Evaluates real-time sensor data quality based on noise, bias drift,
and signal characteristics.
"""

import numpy as np
from typing import Dict


def assess_accelerometer_quality(acc_data: np.ndarray,
                                  window_size: int = 50) -> Dict:
    """
    Assess accelerometer data quality over a window.
    
    Parameters
    ----------
    acc_data : np.ndarray, shape (N, 3)
        Recent accelerometer readings.
    window_size : int
        Analysis window size.
    
    Returns
    -------
    quality : dict
        Quality metrics including noise level, magnitude stability.
    """
    if len(acc_data) < window_size:
        window_size = len(acc_data)
    
    recent = acc_data[-window_size:]
    
    # Noise level (standard deviation)
    noise = np.std(recent, axis=0)
    
    # Magnitude stability
    magnitudes = np.linalg.norm(recent, axis=1)
    mag_std = float(np.std(magnitudes))
    mag_mean = float(np.mean(magnitudes))
    
    # Quality score: lower noise = higher quality
    # Typical good MEMS noise: ~0.03 m/s²
    noise_score = max(0.0, min(1.0, 1.0 - np.mean(noise) / 0.5))
    
    return {
        'noise_level': noise.tolist(),
        'magnitude_mean': mag_mean,
        'magnitude_std': mag_std,
        'quality_score': float(noise_score),
    }


def assess_gyroscope_quality(gyro_data: np.ndarray,
                              window_size: int = 50) -> Dict:
    """
    Assess gyroscope data quality.
    """
    if len(gyro_data) < window_size:
        window_size = len(gyro_data)
    
    recent = gyro_data[-window_size:]
    noise = np.std(recent, axis=0)
    
    noise_score = max(0.0, min(1.0, 1.0 - np.mean(noise) / 0.1))
    
    return {
        'noise_level': noise.tolist(),
        'quality_score': float(noise_score),
    }
