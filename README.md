# README.md — Intelligent Dead Reckoning System

## SIH Problem Statement 168
**AI-ML based Intelligent Dead Reckoning System for Seamless Navigation**

> ⚠️ **ONE-DAY PROTOTYPE** — Uses simulated data for demonstration.
> Not real experimental results.

## Core Innovation

> "Instead of only estimating where the vehicle is, our system
> continuously estimates which sensors can be trusted and adapts
> the navigation solution accordingly."

## Quick Start

```bash
# 1. Create virtual environment
python3.13 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate synthetic dataset (if not present)
python generate_dataset.py

# 4. Run tests
python -m pytest tests/ -v

# 5. Launch dashboard
streamlit run app.py
```

## Features Implemented (v1.0)

| # | Feature | Status |
|---|---------|--------|
| 1 | Dataset loading | ✅ |
| 2 | IMU preprocessing (bias, filtering) | ✅ |
| 3 | Basic dead reckoning | ✅ |
| 4 | GNSS blackout simulation | ✅ |
| 5 | GNSS anomaly detection | ✅ |
| 6 | Road/motion disturbance detection | ✅ |
| 7 | Adaptive sensor confidence | ✅ |
| 8 | Simple weighted fusion | ✅ |
| 9 | Accuracy metrics (RMSE, drift) | ✅ |
| 10 | Trajectory comparison | ✅ |
| 11 | Mobile-style dashboard | ✅ |
| 12 | Stress-test scenarios | ✅ |

## Phase 2 (Future)

| Feature | Status |
|---------|--------|
| 1D CNN velocity model | 📋 Interface ready |
| EKF/UKF | 📋 Interface ready |
| Offline OSM maps | 📋 MapProvider abstraction |
| Map matching | 📋 Planned |
| Android (Samsung S24) | 📋 Architecture ready |
| External IMU | 📋 CommonSensorFrame planned |
| 200 Hz edge processing | 📋 Planned |
| ML sensor quality | 📋 Planned |

## 1-Minute Demo Procedure

1. **Start**: `streamlit run app.py`
2. Click **✅ NORMAL** — Show all sensors healthy, low error
3. Click **📡 GNSS LOSS** — GNSS conf→0, Dead Reckoning mode, error increases
4. Click **🛣️ ROUGH ROAD** — Disturbance detected, IMU conf drops
5. Click **💥 COMBINED** — Both effects, worst-case scenario
6. Open **🔬 Technical View** — Show trajectory, error, confidence plots
7. Point to **Method Comparison** table — Baseline vs Adaptive

## Project Structure

```
NOMAP/
├── app.py                          # Streamlit dashboard
├── requirements.txt                # Dependencies
├── generate_dataset.py             # Synthetic data generator
├── README.md / DATASET.md / MATH.md / ARCHITECTURE.md
├── TESTING.md / RESULTS.md
├── data/sample/                    # Simulated dataset
├── src/
│   ├── data/                       # Loading & preprocessing
│   ├── sensors/                    # Calibration, orientation, transforms
│   ├── navigation/                 # DR, fusion, constraints
│   ├── detection/                  # GNSS anomaly, road disturbance
│   ├── evaluation/                 # Metrics, plots
│   └── visualization/             # Dashboard components
├── models/                         # Phase 2: AI models
└── tests/                          # Unit tests (19 passing)
```

## Tech Stack

- Python 3.13
- NumPy, Pandas, SciPy, scikit-learn
- Plotly (interactive plots)
- Streamlit (dashboard)
- Folium (maps)

## Team

SIH 2024 — Problem Statement 168
