# DATASET.md — Simulated Dataset Documentation

> ⚠️ **SIMULATED DATA — NOT REAL EXPERIMENTAL RESULTS**

## Overview

This prototype uses a synthetically generated dataset that simulates a 90-second
urban drive. The dataset was created specifically for software development and
demonstration purposes.

**No real sensor data or experimental results are used in this prototype.**

## Dataset Location

```
data/sample/simulated_drive.csv
data/sample/metadata.json
```

## Columns

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `timestamp` | float | seconds | Time from start (0 to 90s) |
| `acc_x` | float | m/s² | Accelerometer X (phone forward axis) |
| `acc_y` | float | m/s² | Accelerometer Y (phone lateral axis) |
| `acc_z` | float | m/s² | Accelerometer Z (phone vertical axis, includes gravity ~9.81) |
| `gyro_x` | float | rad/s | Gyroscope X (roll rate) |
| `gyro_y` | float | rad/s | Gyroscope Y (pitch rate) |
| `gyro_z` | float | rad/s | Gyroscope Z (yaw rate) |
| `gnss_lat` | float | degrees | GNSS latitude (WGS84) |
| `gnss_lon` | float | degrees | GNSS longitude (WGS84) |
| `gnss_speed` | float | m/s | GNSS-reported speed |
| `gnss_heading` | float | radians | GNSS-reported heading (0=North) |
| `gnss_hdop` | float | — | Horizontal Dilution of Precision |
| `gnss_num_sats` | int | — | Number of satellites |
| `gnss_fix_quality` | int | — | Fix quality (1=GPS fix) |
| `gt_x` | float | meters | Ground truth East position (local frame) |
| `gt_y` | float | meters | Ground truth North position (local frame) |
| `gt_lat` | float | degrees | Ground truth latitude |
| `gt_lon` | float | degrees | Ground truth longitude |
| `gt_speed` | float | m/s | Ground truth speed |
| `gt_heading` | float | radians | Ground truth heading |
| `gt_acc_x` | float | m/s² | Ground truth forward acceleration |
| `gt_acc_y` | float | m/s² | Ground truth lateral acceleration |
| `gt_acc_z` | float | m/s² | Ground truth vertical acceleration |

## Sampling Rates

- **IMU**: 100 Hz (dt = 0.01 s)
- **GNSS**: 10 Hz (merged onto IMU timestamps via forward-fill)

## Coordinate Convention

- **Local frame**: East-North-Up (ENU)
  - X = East (meters)
  - Y = North (meters)
  - Z = Up (meters)
- **Heading**: 0 = North, π/2 = East (navigation convention)
- **Origin**: Hyderabad, India (17.385044°N, 78.486671°E)

## Simulated Drive Profile

| Time (s) | Segment | Speed | Heading |
|----------|---------|-------|---------|
| 0–5 | Acceleration | 0 → 11 m/s | North |
| 5–20 | Straight | ~11 m/s | North |
| 20–35 | Right turn | ~11 m/s | North → East |
| 35–55 | Straight | ~11 m/s | East |
| 55–65 | Left turn | ~11 m/s | East → NE |
| 65–80 | Deceleration | 11 → 3.5 m/s | NE |
| 80–90 | Slow | ~3.5 m/s | NE |

## Simulated Sensor Characteristics

### IMU Noise
- Accelerometer noise: σ = 0.03 m/s² (typical MEMS)
- Accelerometer bias: [0.05, -0.03, 0.02] m/s²
- Gyroscope noise: σ = 0.005 rad/s
- Gyroscope bias: [0.001, -0.0005, 0.002] rad/s

### GNSS Noise
- Position noise: σ ≈ 3 m (typical open-sky GPS)
- Speed noise: σ = 0.3 m/s
- Heading noise: σ ≈ 2°
- HDOP: uniform [0.8, 1.5]
- Satellites: 8–13

### Road Disturbances
- Injected at t = 42.0, 43.5, 45.0, 47.0, 48.5 seconds
- Spike magnitude: 3–8 m/s² above normal
- Duration: ~50 ms each

## Limitations

1. Synthetic data does not capture real-world IMU sensor complexities
2. No real multipath, urban canyon, or tunnel effects
3. GNSS noise is idealized (Gaussian)
4. Road disturbances are periodic and uniform
5. No magnetometer data
6. No real elevation changes

## For Real Dataset Integration

When IO-VNBD or real sensor data becomes available:
1. Update `src/data/loader.py` column mappings
2. Verify units and coordinate conventions
3. Determine actual sampling frequencies
4. Re-estimate sensor noise characteristics
5. Update DATASET.md with real data findings
