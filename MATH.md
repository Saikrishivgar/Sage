# MATH.md — Mathematical Foundation

All formulas used in the prototype with units documented.

## A. Acceleration Magnitude

```
a_mag = √(ax² + ay² + az²)
```

- **Units**: m/s²
- **Expected at rest**: ≈ 9.81 m/s² (gravity)
- **Used for**: road disturbance detection, sensor quality assessment

## B. Bias Correction

```
a_corrected = a_raw - bias
```

- **bias**: estimated from initial stationary period (first 1 second)
- **Units**: m/s² for accelerometer, rad/s for gyroscope
- **For accelerometer Z**: bias = mean(az_static) - 9.81

## C. Heading Integration

```
θ_k = θ_(k-1) + ω_z · Δt
```

- **θ**: heading angle (radians), 0 = North, π/2 = East
- **ω_z**: yaw rate from gyroscope (rad/s)
- **Δt**: time step (seconds)

## D. Navigation Frame Acceleration

```
a_east  = a_forward · sin(θ) + a_lateral · cos(θ)
a_north = a_forward · cos(θ) - a_lateral · sin(θ)
```

- Transforms body-frame acceleration to ENU navigation frame
- **Units**: m/s²

## E. Velocity Integration (Trapezoidal)

```
v_k = v_(k-1) + a_k · Δt
```

For improved accuracy, trapezoidal integration:

```
p_k = p_(k-1) + 0.5 · (v_(k-1) + v_k) · Δt
```

- **Units**: m/s for velocity, m for position

## F. Position Error

```
e_t = ‖p_estimated,t - p_groundtruth,t‖
    = √((x_est - x_gt)² + (y_est - y_gt)²)
```

- **Units**: meters
- **Calculated at each timestep**

## G. Mean Error

```
mean_error = (1/N) · Σ e_t
```

- **Units**: meters

## H. Root Mean Square Error (RMSE)

```
RMSE = √((1/N) · Σ e_t²)
```

- **Units**: meters
- **Interpretation**: penalizes large errors more than mean error

## I. Drift Percentage

```
drift% = (final_position_error / distance_travelled) × 100
```

- **Units**: percent (%)
- **Target**: < 10% (SIH PS-168)
- **distance_travelled**: sum of inter-point ground-truth distances

## J. Distance Travelled

```
D = Σ √((x_i - x_(i-1))² + (y_i - y_(i-1))²)
```

- **Units**: meters
- Computed along ground truth trajectory

## K. Weighted Fusion

```
p_fused = w_GNSS · p_GNSS + w_INS · p_INS
```

Where weights are derived from confidence:

```
w_GNSS = C_GNSS / (C_GNSS + C_INS)
w_INS  = C_INS / (C_GNSS + C_INS)
```

- **C_GNSS, C_INS**: confidence values ∈ [0, 1]
- During GNSS outage: C_GNSS = 0 → w_GNSS = 0

## L. Coordinate Transforms

### Local Tangent Plane (Flat Earth Approximation)

```
x_meters = (lon - lon₀) × 111132.92 × cos(lat₀)
y_meters = (lat - lat₀) × 111132.92
```

- **Valid for**: distances < ~10 km
- **Frame**: ENU (East-North-Up)

### Inverse Transform

```
lat = lat₀ + y_meters / 111132.92
lon = lon₀ + x_meters / (111132.92 × cos(lat₀))
```

## M. GNSS Anomaly Detection Thresholds

| Check | Threshold | Confidence Factor |
|-------|-----------|-------------------|
| Position jump | > 50 m between fixes | × 0.2 |
| Implied speed | > 55 m/s (~200 km/h) | × 0.3 |
| GNSS speed | > 50 m/s (~180 km/h) | × 0.3 |
| INS disagreement | > 30 m | × 0.5 |
| HDOP > 3.0 | — | × max(0.3, 1 - (hdop-1)×0.15) |

## N. Road Disturbance Thresholds

| Check | Threshold | Severity |
|-------|-----------|----------|
| Spike | deviation > 3.0 m/s² | deviation / 10.0 |
| Pothole-like | deviation > 5.0 m/s² | deviation / 10.0 |
| Rough road | window mean > 1.5 m/s², std > 0.5 | mean / 5.0 |
| Hard braking | a_forward < -3.0 m/s² | |a_forward| / 8.0 |

## O. Sensor Confidence Update Rules

### GNSS Confidence
- **Outage**: C_GNSS -= dt × 2.0 (rapid decay)
- **Anomaly**: C_GNSS *= 0.9
- **Recovery**: C_GNSS → target (exponential approach, rate = dt × 5.0)
- **Target**: min(1.0, 2.0 / HDOP)

### IMU Confidence
- **Disturbance**: C_IMU -= severity × 0.3 × dt × 10
- **Recovery**: C_IMU += dt × 2.0
- **Drift penalty**: if integration_time > 5s, penalty = min(0.3, (t-5) × 0.01) × dt

> **Note**: These are rule-based heuristics for the prototype.
> Phase 2 will replace with ML-based sensor reliability estimation.
