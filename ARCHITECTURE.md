# ARCHITECTURE.md — System Architecture

## System Overview

```
┌──────────────┐     ┌──────────────┐
│   IMU Data   │     │  GNSS Data   │
│  (100 Hz)    │     │  (10 Hz)     │
└──────┬───────┘     └──────┬───────┘
       │                    │
       ▼                    ▼
┌──────────────┐     ┌──────────────┐
│ Preprocessing│     │ GNSS Anomaly │
│ Calibration  │     │  Detector    │
│ Filtering    │     │              │
└──────┬───────┘     └──────┬───────┘
       │                    │
       ▼                    ▼
┌──────────────┐     ┌──────────────┐
│    Dead      │     │   GNSS       │
│  Reckoning   │     │ Confidence   │
│  (Baseline)  │     │  [0,1]       │
└──────┬───────┘     └──────┬───────┘
       │                    │
       ▼                    ▼
┌──────────────────────────────────┐
│       Road Disturbance           │
│         Detector                 │
│     IMU Confidence [0,1]         │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│      ADAPTIVE SENSOR FUSION      │
│                                  │
│  w_GNSS × GNSS + w_INS × INS   │
│                                  │
│  Weights from confidence values  │
└──────────────┬───────────────────┘
               │
        ┌──────┴──────┐
        ▼             ▼
┌────────────┐ ┌────────────┐
│ TRAJECTORY │ │  ACCURACY  │
│    MAP     │ │  METRICS   │
│            │ │  RMSE,     │
│            │ │  Drift %   │
└────────────┘ └────────────┘
```

## Module Dependencies

```
app.py
├── src/data/loader.py
├── src/data/preprocessing.py
├── src/sensors/
│   ├── calibration.py
│   ├── orientation.py
│   ├── coordinate_transform.py
│   └── quality.py
├── src/navigation/
│   ├── dead_reckoning.py
│   ├── fusion.py (SensorConfidence + AdaptiveFusion)
│   └── constraints.py
├── src/detection/
│   ├── gnss_anomaly.py
│   └── road_disturbance.py
├── src/evaluation/
│   ├── metrics.py
│   └── plots.py
└── src/visualization/
    └── dashboard.py
```

## Key Design Principles

### 1. Modular Independence
Each module can be tested and replaced independently.
No circular dependencies.

### 2. Sensor-Agnostic Interfaces
Modules accept numpy arrays, not sensor-specific data structures.
This allows future integration with:
- Android sensor APIs
- External IMU (e.g., Xsens, VectorNav)
- Edge compute platforms

### 3. Transparent Baseline
The dead reckoning baseline is intentionally simple and visible.
Shows the drift problem clearly before adaptive fusion.

### 4. Documented Thresholds
All rule-based thresholds are documented in MATH.md.
Phase 2 will replace with ML-based estimation.

## Future Architecture (Phase 2/3)

```
┌─────────────────────────────────────────────┐
│              SENSOR SOURCES                  │
│                                              │
│  Samsung S24    External IMU    Edge Device   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Acc/Gyro │  │ 200 Hz   │  │ Real-time│  │
│  │ GNSS     │  │ IMU      │  │ 10 Hz    │  │
│  │ Mag      │  │          │  │          │  │
│  └──────────┘  └──────────┘  └──────────┘  │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│         COMMON SENSOR FRAME                  │
│  Unified sensor interface                    │
│  Timestamp synchronization                   │
│  Unit normalization                          │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│         NAVIGATION ENGINE                    │
│                                              │
│  ┌────────┐ ┌────────┐ ┌────────────────┐  │
│  │  DR    │ │  EKF/  │ │ AI Velocity    │  │
│  │Baseline│ │  UKF   │ │ (1D CNN)       │  │
│  └────────┘ └────────┘ └────────────────┘  │
│                                              │
│  ┌────────────────────────────────────────┐  │
│  │  ML SENSOR QUALITY ESTIMATOR          │  │
│  │  ML ROAD CLASSIFIER                   │  │
│  │  ADAPTIVE CONFIDENCE ENGINE           │  │
│  └────────────────────────────────────────┘  │
│                                              │
│  ┌────────┐ ┌────────┐ ┌────────────────┐  │
│  │ Map    │ │ NHC    │ │ Offline OSM    │  │
│  │ Match  │ │        │ │                │  │
│  └────────┘ └────────┘ └────────────────┘  │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│         OUTPUT LAYER                         │
│                                              │
│  Position · Velocity · Heading               │
│  Confidence · Uncertainty                    │
│  Events · Accuracy                           │
└─────────────────────────────────────────────┘
```

## Phase 2 Roadmap

| Feature | Priority | Complexity | Module |
|---------|----------|------------|--------|
| EKF/UKF | High | Medium | `src/navigation/ekf.py` |
| 1D CNN Velocity | High | Medium | `models/velocity_cnn.py` |
| ML Sensor Quality | High | High | `src/sensors/ml_quality.py` |
| Offline OSM | Medium | Medium | `src/map/offline_provider.py` |
| Map Matching | Medium | High | `src/navigation/map_match.py` |
| Android Integration | Medium | High | `android/` |
| External IMU | Low | Low | `src/sensors/external_imu.py` |
| NHC | Low | Medium | `src/navigation/constraints.py` |
| Edge 200 Hz | Low | High | `edge/` |
