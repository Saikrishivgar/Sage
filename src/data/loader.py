"""
Data Loader Module
==================
Loads and validates IMU+GNSS datasets from CSV files.
Supports both the simulated dataset and future real datasets (IO-VNBD, etc.).
"""

import pandas as pd
import numpy as np
import json
import os
from typing import Tuple, Dict, Optional


# Expected column mappings
REQUIRED_IMU_COLS = ['timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
REQUIRED_GNSS_COLS = ['gnss_lat', 'gnss_lon', 'gnss_speed']
GROUND_TRUTH_COLS = ['gt_x', 'gt_y', 'gt_lat', 'gt_lon', 'gt_speed', 'gt_heading']


def load_dataset(filepath: str, metadata_path: Optional[str] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Load a navigation dataset from CSV.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    metadata_path : str, optional
        Path to metadata JSON. If None, tries to find it next to the CSV.
    
    Returns
    -------
    df : pd.DataFrame
        The loaded and validated dataset.
    meta : dict
        Dataset metadata (sampling rates, origin, etc.).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Validate required columns
    missing_imu = [c for c in REQUIRED_IMU_COLS if c not in df.columns]
    if missing_imu:
        raise ValueError(f"Missing required IMU columns: {missing_imu}")
    
    missing_gnss = [c for c in REQUIRED_GNSS_COLS if c not in df.columns]
    if missing_gnss:
        print(f"Warning: Missing GNSS columns: {missing_gnss}")
    
    has_ground_truth = all(c in df.columns for c in GROUND_TRUTH_COLS)
    
    # Load metadata
    meta = {}
    if metadata_path is None:
        metadata_path = os.path.join(os.path.dirname(filepath), 'metadata.json')
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            meta = json.load(f)
    else:
        # Infer metadata
        meta = infer_metadata(df)
    
    meta['has_ground_truth'] = has_ground_truth
    meta['filepath'] = filepath
    meta['n_samples'] = len(df)
    
    return df, meta


def infer_metadata(df: pd.DataFrame) -> Dict:
    """Infer dataset metadata from the data itself."""
    timestamps = df['timestamp'].values
    dt = np.median(np.diff(timestamps))
    
    meta = {
        'duration': timestamps[-1] - timestamps[0],
        'imu_rate': int(round(1.0 / dt)),
        'dt': float(dt),
    }
    
    # Try to find origin lat/lon
    if 'gt_lat' in df.columns:
        meta['lat0'] = float(df['gt_lat'].iloc[0])
        meta['lon0'] = float(df['gt_lon'].iloc[0])
        meta['m_per_deg_lat'] = 111132.92
        meta['m_per_deg_lon'] = 111132.92 * np.cos(np.radians(meta['lat0']))
    
    return meta


def get_dataset_summary(df: pd.DataFrame, meta: Dict) -> str:
    """Generate a human-readable dataset summary."""
    lines = [
        "=" * 50,
        "DATASET SUMMARY",
        "=" * 50,
        f"File: {meta.get('filepath', 'unknown')}",
        f"Samples: {meta.get('n_samples', len(df))}",
        f"Duration: {meta.get('duration', 0):.1f} seconds",
        f"IMU Rate: {meta.get('imu_rate', 'unknown')} Hz",
        f"Has Ground Truth: {meta.get('has_ground_truth', False)}",
        "",
        "Columns:",
    ]
    for col in df.columns:
        lines.append(f"  {col}: {df[col].dtype} | range [{df[col].min():.4f}, {df[col].max():.4f}]")
    
    return "\n".join(lines)
