# TESTING.md — Test Plan & Results

## Test Suite

Located at: `tests/test_navigation.py`

Run with:
```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

## Test Results

```
19 passed in 0.11s
```

| # | Test | Category | Result |
|---|------|----------|--------|
| 1 | `test_normal_dr_initialization` | Normal | ✅ PASS |
| 2 | `test_normal_straight_drive` | Normal | ✅ PASS |
| 3 | `test_gnss_confidence_drops_to_zero` | GNSS Blackout | ✅ PASS |
| 4 | `test_gnss_confidence_recovers` | GNSS Blackout | ✅ PASS |
| 5 | `test_position_jump_detected` | GNSS Anomaly | ✅ PASS |
| 6 | `test_normal_gnss_not_flagged` | GNSS Anomaly | ✅ PASS |
| 7 | `test_normal_not_detected` | Rough Road | ✅ PASS |
| 8 | `test_pothole_detected` | Rough Road | ✅ PASS |
| 9 | `test_gnss_confidence_bounded` | Confidence | ✅ PASS |
| 10 | `test_imu_confidence_bounded` | Confidence | ✅ PASS |
| 11 | `test_zero_drift` | Drift Calc | ✅ PASS |
| 12 | `test_known_drift` | Drift Calc | ✅ PASS |
| 13 | `test_zero_distance` | Drift Calc | ✅ PASS |
| 14 | `test_zero_error` | Position Error | ✅ PASS |
| 15 | `test_known_error` | Position Error | ✅ PASS |
| 16 | `test_rmse_calculation` | Position Error | ✅ PASS |
| 17 | `test_full_metrics` | Position Error | ✅ PASS |
| 18 | `test_roundtrip` | Coordinates | ✅ PASS |
| 19 | `test_origin_is_zero` | Coordinates | ✅ PASS |

## Test Coverage

| Module | Tested |
|--------|--------|
| Dead Reckoning Engine | ✅ |
| Sensor Confidence | ✅ |
| Adaptive Fusion | ✅ (via confidence) |
| GNSS Anomaly Detection | ✅ |
| Road Disturbance Detection | ✅ |
| Accuracy Metrics | ✅ |
| Coordinate Transforms | ✅ |
| Data Loader | ⬜ (integration tested via app) |
| Preprocessing | ⬜ (integration tested via app) |
| Dashboard UI | ⬜ (manual testing) |

## Manual Test Plan

| # | Scenario | Steps | Expected |
|---|----------|-------|----------|
| 1 | Normal | Click NORMAL button | All confidence high, low error |
| 2 | GNSS Loss | Click GNSS LOSS button | GNSS conf→0, DR mode, trajectory diverges |
| 3 | Rough Road | Click ROUGH ROAD button | IMU conf drops, disturbance events |
| 4 | GNSS Jump | Click GNSS JUMP button | Anomaly detected, GNSS conf drops |
| 5 | Combined | Click COMBINED button | Both GNSS and IMU confidence affected |
| 6 | Custom | Set custom blackout params | Pipeline runs with custom settings |
