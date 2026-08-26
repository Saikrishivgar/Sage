"""
Test Suite for Intelligent Dead Reckoning System
==================================================
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.navigation.dead_reckoning import DeadReckoningEngine, run_dead_reckoning
from src.navigation.fusion import SensorConfidence, AdaptiveFusion
from src.detection.gnss_anomaly import GNSSAnomalyDetector
from src.detection.road_disturbance import RoadDisturbanceDetector
from src.evaluation.metrics import (
    position_error, mean_error, rmse, max_error,
    drift_percentage, distance_travelled, compute_full_metrics
)
from src.sensors.coordinate_transform import latlon_to_local, local_to_latlon


class TestNormal:
    """Test normal operation with GNSS + INS."""
    
    def test_normal_dr_initialization(self):
        """DR engine initializes with correct state."""
        engine = DeadReckoningEngine(
            initial_position=np.array([0.0, 0.0]),
            initial_velocity=np.array([0.0, 0.0]),
            initial_heading=0.0,
        )
        assert np.allclose(engine.position, [0.0, 0.0])
        assert np.allclose(engine.velocity, [0.0, 0.0])
        assert engine.heading == 0.0
    
    def test_normal_straight_drive(self):
        """DR produces forward motion for constant forward acceleration."""
        engine = DeadReckoningEngine(
            initial_position=np.array([0.0, 0.0]),
            initial_heading=0.0,  # North
        )
        # 1 second at 1 m/s² forward
        for _ in range(100):
            engine.update(acc_forward=1.0, acc_lateral=0.0, gyro_z=0.0, dt=0.01)
        
        traj = engine.get_trajectory()
        # Should have moved north (positive y)
        assert traj['y'][-1] > 0
        assert abs(traj['x'][-1]) < 1.0  # Minimal east drift


class TestGNSSBlackout:
    """Test GNSS blackout scenario."""
    
    def test_gnss_confidence_drops_to_zero(self):
        """GNSS confidence reaches 0 during outage."""
        conf = SensorConfidence()
        
        # Simulate 5 seconds of outage
        for _ in range(500):
            conf.update_gnss_confidence(gnss_available=False, dt=0.01)
        
        assert conf.gnss_confidence < 0.01
    
    def test_gnss_confidence_recovers(self):
        """GNSS confidence recovers after outage ends."""
        conf = SensorConfidence()
        
        # Outage
        for _ in range(500):
            conf.update_gnss_confidence(gnss_available=False, dt=0.01)
        assert conf.gnss_confidence < 0.01
        
        # Recovery
        for _ in range(500):
            conf.update_gnss_confidence(gnss_available=True, hdop=1.0, dt=0.01)
        assert conf.gnss_confidence > 0.5


class TestGNSSAnomaly:
    """Test GNSS anomaly detection."""
    
    def test_position_jump_detected(self):
        """Large position jump is detected as anomaly."""
        detector = GNSSAnomalyDetector()
        
        lat0, lon0 = 17.385, 78.487
        
        # First normal fix
        detector.detect(17.385, 78.487, 10.0, 1.0, 0, 0, 0.0, lat0, lon0)
        
        # Jump: ~55m north
        result = detector.detect(17.3855, 78.487, 10.0, 1.0, 0, 0, 0.1, lat0, lon0)
        
        assert result['anomaly_detected'] == True
        assert result['gnss_confidence'] < 0.5
    
    def test_normal_gnss_not_flagged(self):
        """Normal GNSS progression is not flagged."""
        detector = GNSSAnomalyDetector()
        
        lat0, lon0 = 17.385, 78.487
        
        detector.detect(17.385000, 78.487000, 10.0, 1.0, 0, 0, 0.0, lat0, lon0)
        result = detector.detect(17.385001, 78.487001, 10.0, 1.0, 1, 1, 0.1, lat0, lon0)
        
        assert result['anomaly_detected'] == False
        assert result['gnss_confidence'] > 0.8


class TestRoughRoad:
    """Test road disturbance detection."""
    
    def test_normal_not_detected(self):
        """Normal driving not flagged as disturbance."""
        detector = RoadDisturbanceDetector(fs=100.0)
        
        # Normal: gravity only, minor noise
        result = detector.detect(0.05, -0.03, 9.82, 0.0, 0.0)
        assert result['detected'] == False
        assert result['classification'] == 'NORMAL'
    
    def test_pothole_detected(self):
        """Large acceleration spike detected as disturbance."""
        detector = RoadDisturbanceDetector(fs=100.0)
        
        # Feed some normal samples first
        for i in range(50):
            detector.detect(0.05, -0.03, 9.81, 0.0, i * 0.01)
        
        # Pothole-like spike
        result = detector.detect(2.0, 1.0, 16.0, -1.0, 0.5)
        assert result['detected'] == True
        assert result['severity'] > 0.0


class TestConfidenceRange:
    """Test confidence values stay in [0, 1]."""
    
    def test_gnss_confidence_bounded(self):
        """GNSS confidence always in [0, 1]."""
        conf = SensorConfidence()
        
        # Extreme scenarios
        for _ in range(10000):
            conf.update_gnss_confidence(gnss_available=False, dt=0.01)
        assert 0.0 <= conf.gnss_confidence <= 1.0
        
        for _ in range(10000):
            conf.update_gnss_confidence(gnss_available=True, hdop=0.5, dt=0.01)
        assert 0.0 <= conf.gnss_confidence <= 1.0
    
    def test_imu_confidence_bounded(self):
        """IMU confidence always in [0, 1]."""
        conf = SensorConfidence()
        
        for _ in range(10000):
            conf.update_imu_confidence(disturbance_detected=True,
                                        disturbance_severity=1.0, dt=0.01)
        assert 0.0 <= conf.imu_confidence <= 1.0
        
        for _ in range(10000):
            conf.update_imu_confidence(disturbance_detected=False, dt=0.01)
        assert 0.0 <= conf.imu_confidence <= 1.0


class TestDriftCalculation:
    """Test drift percentage calculation."""
    
    def test_zero_drift(self):
        """Perfect tracking has zero drift."""
        x = np.array([0, 1, 2, 3, 4, 5], dtype=float)
        y = np.array([0, 0, 0, 0, 0, 0], dtype=float)
        
        dist = distance_travelled(x, y)
        assert dist == pytest.approx(5.0)
        
        drift = drift_percentage(0.0, dist)
        assert drift == 0.0
    
    def test_known_drift(self):
        """Known error produces correct drift."""
        dist = 100.0  # 100 meters
        error = 5.0   # 5 meters
        
        drift = drift_percentage(error, dist)
        assert drift == pytest.approx(5.0)
    
    def test_zero_distance(self):
        """Zero distance doesn't cause division by zero."""
        drift = drift_percentage(1.0, 0.0)
        assert drift == 0.0


class TestPositionError:
    """Test position error calculation."""
    
    def test_zero_error(self):
        """Identical trajectories have zero error."""
        x = np.array([0, 1, 2, 3], dtype=float)
        y = np.array([0, 0, 0, 0], dtype=float)
        
        errors = position_error(x, y, x, y)
        assert np.allclose(errors, 0.0)
    
    def test_known_error(self):
        """Known offset produces correct error."""
        x_est = np.array([3.0, 3.0])
        y_est = np.array([4.0, 4.0])
        x_gt = np.array([0.0, 0.0])
        y_gt = np.array([0.0, 0.0])
        
        errors = position_error(x_est, y_est, x_gt, y_gt)
        assert np.allclose(errors, 5.0)  # 3-4-5 triangle
    
    def test_rmse_calculation(self):
        """RMSE computed correctly."""
        errors = np.array([1.0, 2.0, 3.0])
        result = rmse(errors)
        expected = np.sqrt(np.mean(errors**2))
        assert result == pytest.approx(expected)
    
    def test_full_metrics(self):
        """Full metrics computation works."""
        x_est = np.array([0, 1, 2, 3, 4], dtype=float)
        y_est = np.array([0, 0.1, 0.2, 0.3, 0.4], dtype=float)
        x_gt = np.array([0, 1, 2, 3, 4], dtype=float)
        y_gt = np.array([0, 0, 0, 0, 0], dtype=float)
        
        metrics = compute_full_metrics(x_est, y_est, x_gt, y_gt)
        
        assert 'mean_error_m' in metrics
        assert 'rmse_m' in metrics
        assert 'max_error_m' in metrics
        assert 'drift_percent' in metrics
        assert 'pass_fail' in metrics
        assert metrics['mean_error_m'] >= 0
        assert metrics['rmse_m'] >= 0


class TestCoordinateTransform:
    """Test coordinate transformations."""
    
    def test_roundtrip(self):
        """Local to latlon and back preserves position."""
        lat0, lon0 = 17.385, 78.487
        x_orig = np.array([100.0, 200.0])
        y_orig = np.array([50.0, 150.0])
        
        lat, lon = local_to_latlon(x_orig, y_orig, lat0, lon0)
        x_back, y_back = latlon_to_local(lat, lon, lat0, lon0)
        
        assert np.allclose(x_orig, x_back, atol=0.01)
        assert np.allclose(y_orig, y_back, atol=0.01)
    
    def test_origin_is_zero(self):
        """Origin maps to (0, 0) in local frame."""
        lat0, lon0 = 17.385, 78.487
        x, y = latlon_to_local(np.array([lat0]), np.array([lon0]), lat0, lon0)
        assert np.allclose(x, 0.0)
        assert np.allclose(y, 0.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
