"""
Data Preprocessing Module
==========================
Handles: timestamp sorting, dedup, missing values, resampling,
bias estimation, filtering, normalization.
"""

import numpy as np
import pandas as pd
from scipy import signal
from typing import Tuple, Dict


def preprocess_imu(df: pd.DataFrame, meta: Dict) -> Tuple[pd.DataFrame, Dict]:
    """
    Full preprocessing pipeline for IMU data.
    
    Steps:
    1. Sort by timestamp
    2. Remove duplicates
    3. Handle missing values
    4. Estimate and remove static bias
    5. Apply low-pass filter (gentle, not aggressive)
    
    Returns
    -------
    df_processed : pd.DataFrame
        Preprocessed data with original columns preserved as *_raw.
    preprocessing_info : dict
        Bias estimates, filter parameters, etc.
    """
    df = df.copy()
    info = {}
    
    # 1. Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 2. Remove duplicates
    n_before = len(df)
    df = df.drop_duplicates(subset='timestamp').reset_index(drop=True)
    info['duplicates_removed'] = n_before - len(df)
    
    # 3. Handle missing values (forward fill then backward fill)
    imu_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    n_missing = df[imu_cols].isna().sum().sum()
    df[imu_cols] = df[imu_cols].ffill().bfill()
    info['missing_values_filled'] = int(n_missing)
    
    # 4. Estimate static bias from first 1 second (assuming vehicle is stationary)
    dt = meta.get('dt', 0.01)
    n_static = min(int(1.0 / dt), len(df) // 10)
    
    # Save raw values
    for col in imu_cols:
        df[f'{col}_raw'] = df[col].copy()
    
    # Accelerometer bias (subtract gravity from z)
    acc_bias = {
        'acc_x': float(df['acc_x'].iloc[:n_static].mean()),
        'acc_y': float(df['acc_y'].iloc[:n_static].mean()),
        'acc_z': float(df['acc_z'].iloc[:n_static].mean() - 9.81),  # Remove gravity
    }
    
    # Gyroscope bias
    gyro_bias = {
        'gyro_x': float(df['gyro_x'].iloc[:n_static].mean()),
        'gyro_y': float(df['gyro_y'].iloc[:n_static].mean()),
        'gyro_z': float(df['gyro_z'].iloc[:n_static].mean()),
    }
    
    info['acc_bias'] = acc_bias
    info['gyro_bias'] = gyro_bias
    info['n_static_samples'] = n_static
    
    # Apply bias correction
    for col, bias in {**acc_bias, **gyro_bias}.items():
        df[col] = df[col] - bias
    
    # 5. Gentle low-pass filter (20 Hz cutoff for 100 Hz data — keeps motion, removes high-freq noise)
    fs = meta.get('imu_rate', 100)
    cutoff = 20.0  # Hz
    
    if fs > 2 * cutoff:
        nyq = fs / 2.0
        b, a = signal.butter(2, cutoff / nyq, btype='low')
        
        for col in imu_cols:
            df[f'{col}_filtered'] = signal.filtfilt(b, a, df[col].values)
        
        info['filter'] = {
            'type': 'butterworth',
            'order': 2,
            'cutoff_hz': cutoff,
            'fs_hz': fs,
        }
    
    # Compute acceleration magnitude
    df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    df['acc_mag_raw'] = np.sqrt(df['acc_x_raw']**2 + df['acc_y_raw']**2 + df['acc_z_raw']**2)
    
    return df, info


def compute_dt(df: pd.DataFrame) -> np.ndarray:
    """Compute time differences between samples."""
    dt = np.diff(df['timestamp'].values, prepend=df['timestamp'].values[0] - 0.01)
    dt[0] = dt[1]  # Fix first element
    return dt
