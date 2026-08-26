"""
Synthetic Dataset Generator for Dead Reckoning Prototype
=========================================================
SIMULATED DATA — NOT REAL EXPERIMENTAL RESULTS

Generates a realistic 90-second urban drive with:
- 100 Hz IMU (accelerometer + gyroscope)
- 10 Hz GNSS (lat, lon, speed, heading)
- Ground truth trajectory
- Realistic sensor noise, bias, and disturbances
"""

import numpy as np
import pandas as pd
import os


def generate_drive_trajectory(duration=90.0, dt=0.01):
    """
    Generate a ground-truth urban drive trajectory.
    
    The drive consists of segments:
    - 0-20s:  Straight acceleration to ~40 km/h (~11 m/s)
    - 20-35s: Gentle right turn
    - 35-55s: Straight at ~40 km/h
    - 55-65s: Left turn
    - 65-80s: Straight, decelerating
    - 80-90s: Slow straight
    
    Returns arrays in local tangent plane (ENU):
    - X = East (meters)
    - Y = North (meters)
    """
    n_samples = int(duration / dt)
    t = np.arange(n_samples) * dt
    
    # Ground truth velocity profile (m/s)
    speed = np.zeros(n_samples)
    heading = np.zeros(n_samples)  # radians, 0 = North, clockwise positive
    
    # Build velocity profile
    for i in range(n_samples):
        ti = t[i]
        if ti < 5:
            speed[i] = ti * 2.2  # Accelerate 0 to 11 m/s
        elif ti < 20:
            speed[i] = 11.0
        elif ti < 35:
            speed[i] = 11.0  # Turning at constant speed
        elif ti < 55:
            speed[i] = 11.0
        elif ti < 65:
            speed[i] = 11.0  # Turning
        elif ti < 80:
            speed[i] = 11.0 - (ti - 65) * 0.5  # Decelerate
        else:
            speed[i] = max(3.5, 11.0 - (ti - 65) * 0.5)
    
    # Build heading profile (radians)
    # Start heading North (pi/2 in math convention, but we use navigation convention)
    current_heading = 0.0  # North
    for i in range(n_samples):
        ti = t[i]
        if 20 <= ti < 35:
            # Right turn: heading changes by ~90 degrees over 15 seconds
            current_heading = (ti - 20) / 15.0 * (np.pi / 2)
        elif 35 <= ti < 55:
            current_heading = np.pi / 2  # Heading East
        elif 55 <= ti < 65:
            # Left turn: heading changes by ~45 degrees
            current_heading = np.pi / 2 - (ti - 55) / 10.0 * (np.pi / 4)
        elif ti >= 65:
            current_heading = np.pi / 4  # Heading NE
        heading[i] = current_heading
    
    # Integrate position (navigation frame: X=East, Y=North)
    x = np.zeros(n_samples)
    y = np.zeros(n_samples)
    
    # Starting position: Hyderabad area (example)
    lat0 = 17.385044
    lon0 = 78.486671
    
    for i in range(1, n_samples):
        # heading: 0=North, pi/2=East
        vx = speed[i] * np.sin(heading[i])  # East component
        vy = speed[i] * np.cos(heading[i])  # North component
        x[i] = x[i-1] + vx * dt
        y[i] = y[i-1] + vy * dt
    
    # True accelerations in body frame
    # Forward acceleration = d(speed)/dt
    ax_forward = np.gradient(speed, dt)
    
    # Lateral acceleration from turning (centripetal)
    heading_rate = np.gradient(heading, dt)
    ay_lateral = speed * heading_rate
    
    # Vertical: gravity component (phone assumed flat, z points up)
    az_vertical = np.full(n_samples, 9.81)  # gravity
    
    # Gyroscope: rotation rates
    gx = np.zeros(n_samples)  # roll rate (minimal for car)
    gy = np.zeros(n_samples)  # pitch rate (minimal)
    gz = heading_rate  # yaw rate
    
    return {
        't': t,
        'speed': speed,
        'heading': heading,
        'x': x,
        'y': y,
        'lat0': lat0,
        'lon0': lon0,
        'ax_true': ax_forward,
        'ay_true': ay_lateral,
        'az_true': az_vertical,
        'gx_true': gx,
        'gy_true': gy,
        'gz_true': gz,
        'heading_rate': heading_rate,
    }


def add_imu_noise(truth, seed=42):
    """Add realistic IMU sensor noise and bias."""
    rng = np.random.RandomState(seed)
    n = len(truth['t'])
    
    # Accelerometer noise (typical MEMS: 0.01-0.05 m/s²)
    accel_noise_std = 0.03  # m/s²
    accel_bias = np.array([0.05, -0.03, 0.02])  # m/s² static bias
    
    # Gyroscope noise (typical MEMS: 0.001-0.01 rad/s)
    gyro_noise_std = 0.005  # rad/s
    gyro_bias = np.array([0.001, -0.0005, 0.002])  # rad/s
    
    ax = truth['ax_true'] + accel_bias[0] + rng.normal(0, accel_noise_std, n)
    ay = truth['ay_true'] + accel_bias[1] + rng.normal(0, accel_noise_std, n)
    az = truth['az_true'] + accel_bias[2] + rng.normal(0, accel_noise_std, n)
    
    gx = truth['gx_true'] + gyro_bias[0] + rng.normal(0, gyro_noise_std, n)
    gy = truth['gy_true'] + gyro_bias[1] + rng.normal(0, gyro_noise_std, n)
    gz = truth['gz_true'] + gyro_bias[2] + rng.normal(0, gyro_noise_std, n)
    
    return ax, ay, az, gx, gy, gz


def add_road_disturbances(ax, ay, az, t, seed=43):
    """Add road disturbance spikes at specific time intervals."""
    rng = np.random.RandomState(seed)
    ax_dist = ax.copy()
    ay_dist = ay.copy()
    az_dist = az.copy()
    
    # Add disturbances in the 40-50s window
    disturbance_times = [42.0, 43.5, 45.0, 47.0, 48.5]
    
    for td in disturbance_times:
        idx = np.argmin(np.abs(t - td))
        width = int(0.05 / 0.01)  # 50ms spike
        spike_mag = rng.uniform(3.0, 8.0)
        
        for j in range(max(0, idx - width), min(len(t), idx + width)):
            factor = 1.0 - abs(j - idx) / width
            az_dist[j] += spike_mag * factor
            ax_dist[j] += rng.uniform(-1.0, 1.0) * factor
            ay_dist[j] += rng.uniform(-1.0, 1.0) * factor
    
    return ax_dist, ay_dist, az_dist


def generate_gnss(truth, dt_gnss=0.1, seed=44):
    """Generate 10 Hz GNSS measurements with realistic noise."""
    rng = np.random.RandomState(seed)
    dt_imu = truth['t'][1] - truth['t'][0]
    step = int(dt_gnss / dt_imu)
    
    indices = np.arange(0, len(truth['t']), step)
    
    # Convert local XY to lat/lon
    lat0 = truth['lat0']
    lon0 = truth['lon0']
    
    # Meters per degree (approximate at this latitude)
    m_per_deg_lat = 111132.92
    m_per_deg_lon = 111132.92 * np.cos(np.radians(lat0))
    
    gnss_records = []
    for idx in indices:
        t = truth['t'][idx]
        
        # True position in lat/lon
        true_lat = lat0 + truth['y'][idx] / m_per_deg_lat
        true_lon = lon0 + truth['x'][idx] / m_per_deg_lon
        
        # GNSS noise (typical: 2-5m CEP)
        lat_noise = rng.normal(0, 3.0) / m_per_deg_lat  # ~3m std
        lon_noise = rng.normal(0, 3.0) / m_per_deg_lon
        
        speed_noise = rng.normal(0, 0.3)  # m/s
        heading_noise = rng.normal(0, np.radians(2))  # ~2 deg
        
        gnss_records.append({
            'timestamp': t,
            'gnss_lat': true_lat + lat_noise,
            'gnss_lon': true_lon + lon_noise,
            'gnss_speed': max(0, truth['speed'][idx] + speed_noise),
            'gnss_heading': truth['heading'][idx] + heading_noise,
            'gnss_hdop': rng.uniform(0.8, 1.5),
            'gnss_num_sats': rng.randint(8, 14),
            'gnss_fix_quality': 1,
            # Ground truth
            'gt_lat': true_lat,
            'gt_lon': true_lon,
            'gt_speed': truth['speed'][idx],
            'gt_heading': truth['heading'][idx],
        })
    
    return pd.DataFrame(gnss_records)


def generate_full_dataset(duration=90.0, seed=42):
    """Generate the complete simulated dataset."""
    
    truth = generate_drive_trajectory(duration=duration)
    
    # IMU with noise
    ax, ay, az, gx, gy, gz = add_imu_noise(truth, seed=seed)
    
    # Add road disturbances
    ax, ay, az = add_road_disturbances(ax, ay, az, truth['t'], seed=seed+1)
    
    # Build IMU DataFrame (100 Hz)
    imu_df = pd.DataFrame({
        'timestamp': truth['t'],
        'acc_x': ax,
        'acc_y': ay,
        'acc_z': az,
        'gyro_x': gx,
        'gyro_y': gy,
        'gyro_z': gz,
        # Ground truth in local frame
        'gt_x': truth['x'],
        'gt_y': truth['y'],
        'gt_speed': truth['speed'],
        'gt_heading': truth['heading'],
        'gt_acc_x': truth['ax_true'],
        'gt_acc_y': truth['ay_true'],
        'gt_acc_z': truth['az_true'],
    })
    
    # GNSS (10 Hz)
    gnss_df = generate_gnss(truth, seed=seed+2)
    
    # Merge: IMU is primary, GNSS is sampled at lower rate
    # Create merged dataset with GNSS columns filled forward
    lat0 = truth['lat0']
    lon0 = truth['lon0']
    m_per_deg_lat = 111132.92
    m_per_deg_lon = 111132.92 * np.cos(np.radians(lat0))
    
    # Add lat/lon ground truth to IMU
    imu_df['gt_lat'] = lat0 + truth['y'] / m_per_deg_lat
    imu_df['gt_lon'] = lon0 + truth['x'] / m_per_deg_lon
    
    # Merge GNSS onto IMU timestamps
    merged = pd.merge_asof(
        imu_df.sort_values('timestamp'),
        gnss_df[['timestamp', 'gnss_lat', 'gnss_lon', 'gnss_speed', 
                  'gnss_heading', 'gnss_hdop', 'gnss_num_sats', 'gnss_fix_quality']].sort_values('timestamp'),
        on='timestamp',
        direction='backward',
        tolerance=0.15  # Max 150ms staleness
    )
    
    return merged, {
        'lat0': lat0,
        'lon0': lon0,
        'duration': duration,
        'imu_rate': 100,
        'gnss_rate': 10,
        'm_per_deg_lat': m_per_deg_lat,
        'm_per_deg_lon': m_per_deg_lon,
    }


def save_dataset(output_dir='data/sample'):
    """Generate and save the dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    df, meta = generate_full_dataset()
    
    filepath = os.path.join(output_dir, 'simulated_drive.csv')
    df.to_csv(filepath, index=False)
    
    # Save metadata
    import json
    meta_path = os.path.join(output_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"Dataset saved: {filepath}")
    print(f"Metadata saved: {meta_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Duration: {meta['duration']}s")
    print(f"IMU rate: {meta['imu_rate']} Hz")
    print(f"GNSS rate: {meta['gnss_rate']} Hz")
    
    return df, meta


if __name__ == '__main__':
    save_dataset()
