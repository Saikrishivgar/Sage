"""
Adaptive Sensor Fusion Module
===============================
Weighted fusion of GNSS and INS estimates based on adaptive
sensor confidence values.

Core innovation:
    "Instead of blindly trusting a sensor, the system continuously
    estimates sensor reliability and adapts the navigation solution."

Prototype confidence engine — planned to be replaced by
ML-based sensor reliability estimation in Phase 2.
"""

import numpy as np
from typing import Dict, Optional, List


class SensorConfidence:
    """
    Manages confidence values for all sensors.
    
    All confidence values are in [0.0, 1.0].
    
    Confidence is affected by:
    - GNSS: outage status, anomaly detection, HDOP
    - IMU: road disturbance, integration duration, noise level
    - Map: (Phase 2) map matching quality
    """
    
    def __init__(self):
        self.gnss_confidence = 1.0
        self.imu_confidence = 1.0
        self.map_confidence = 0.5  # Phase 2
        self.overall_confidence = 1.0
        
        # History
        self.history = {
            'gnss': [],
            'imu': [],
            'map': [],
            'overall': [],
        }
        
        # State tracking
        self.gnss_outage_duration = 0.0
        self.imu_disturbance_active = False
        self.gnss_anomaly_active = False
    
    def update_gnss_confidence(self, gnss_available: bool,
                                gnss_anomaly: bool = False,
                                hdop: float = 1.0,
                                dt: float = 0.01):
        """
        Update GNSS confidence based on availability and quality.
        
        Rules (documented — to be replaced by ML in Phase 2):
        - Outage: confidence decays to 0
        - Anomaly: confidence reduced proportionally
        - Good fix: confidence recovers
        - HDOP > 3: confidence reduces
        """
        if not gnss_available:
            # GNSS outage: rapid decay to 0
            self.gnss_outage_duration += dt
            self.gnss_confidence = max(0.0, self.gnss_confidence - dt * 5.0)
        elif gnss_anomaly:
            # GNSS anomaly: reduce confidence
            self.gnss_anomaly_active = True
            self.gnss_confidence = max(0.1, self.gnss_confidence * 0.85)
        else:
            # Good GNSS: recover confidence quickly
            self.gnss_outage_duration = 0.0
            self.gnss_anomaly_active = False
            hdop_factor = min(1.0, 2.0 / max(hdop, 0.5))  # Lower HDOP = higher confidence
            target = hdop_factor
            # Fast recovery: ~0.5s to reach near-target at 100Hz
            self.gnss_confidence = self.gnss_confidence + (target - self.gnss_confidence) * min(1.0, dt * 20.0)
        
        self.gnss_confidence = np.clip(self.gnss_confidence, 0.0, 1.0)
    
    def update_imu_confidence(self, disturbance_detected: bool,
                               disturbance_severity: float = 0.0,
                               integration_time: float = 0.0,
                               dt: float = 0.01):
        """
        Update IMU confidence based on disturbance and drift accumulation.
        
        Rules (documented — to be replaced by ML in Phase 2):
        - Road disturbance: temporary confidence reduction
        - Long integration without GNSS: gradual drift-related reduction
        - Normal: high confidence
        """
        if disturbance_detected:
            # Road disturbance: reduce confidence based on severity
            self.imu_disturbance_active = True
            reduction = disturbance_severity * 0.3  # Max 30% reduction for max severity
            self.imu_confidence = max(0.3, self.imu_confidence - reduction * dt * 10)
        else:
            self.imu_disturbance_active = False
            # Recover IMU confidence
            self.imu_confidence = min(1.0, self.imu_confidence + dt * 2.0)
        
        # Drift accumulation penalty (longer integration = less trust)
        if integration_time > 5.0:
            drift_penalty = min(0.3, (integration_time - 5.0) * 0.01)
            self.imu_confidence = max(0.3, self.imu_confidence - drift_penalty * dt)
        
        self.imu_confidence = np.clip(self.imu_confidence, 0.0, 1.0)
    
    def compute_overall(self) -> float:
        """Compute overall system confidence."""
        # Weighted combination
        if self.gnss_confidence > 0.1:
            self.overall_confidence = 0.6 * self.gnss_confidence + 0.4 * self.imu_confidence
        else:
            # GNSS denied: overall depends heavily on IMU
            self.overall_confidence = 0.3 * self.imu_confidence
        
        self.overall_confidence = np.clip(self.overall_confidence, 0.0, 1.0)
        return self.overall_confidence
    
    def record(self):
        """Record current confidence to history."""
        self.history['gnss'].append(self.gnss_confidence)
        self.history['imu'].append(self.imu_confidence)
        self.history['map'].append(self.map_confidence)
        self.compute_overall()
        self.history['overall'].append(self.overall_confidence)
    
    def get_navigation_mode(self) -> str:
        """Determine current navigation mode."""
        if self.gnss_confidence > 0.7 and self.imu_confidence > 0.7:
            return "GNSS + INS"
        elif self.gnss_confidence > 0.3:
            if self.imu_disturbance_active:
                return "GNSS + INS (DEGRADED)"
            return "GNSS + INS"
        elif self.gnss_confidence < 0.1:
            if self.imu_disturbance_active:
                return "DEAD RECKONING (DEGRADED)"
            return "DEAD RECKONING"
        else:
            return "ADAPTIVE FUSION"
    
    def get_status(self) -> Dict:
        """Get current status summary."""
        return {
            'gnss_confidence': self.gnss_confidence,
            'imu_confidence': self.imu_confidence,
            'map_confidence': self.map_confidence,
            'overall_confidence': self.overall_confidence,
            'navigation_mode': self.get_navigation_mode(),
            'gnss_outage_duration': self.gnss_outage_duration,
            'gnss_anomaly': self.gnss_anomaly_active,
            'imu_disturbance': self.imu_disturbance_active,
        }


class AdaptiveFusion:
    """
    Weighted fusion of GNSS and INS estimates.
    
    final_estimate = w_gnss * gnss_estimate + w_ins * ins_estimate
    
    Weights are derived from confidence values:
    - Good GNSS: GNSS receives greater weight
    - GNSS outage: GNSS weight = 0, INS only
    - Rough road: INS weight decreases
    - GNSS anomaly: GNSS weight decreases
    """
    
    def __init__(self):
        self.confidence = SensorConfidence()
        self.events: List[Dict] = []
    
    def fuse_position(self, gnss_position: Optional[np.ndarray],
                      ins_position: np.ndarray,
                      gnss_available: bool,
                      gnss_anomaly: bool = False,
                      disturbance_detected: bool = False,
                      disturbance_severity: float = 0.0,
                      hdop: float = 1.0,
                      integration_time: float = 0.0,
                      dt: float = 0.01) -> Dict:
        """
        Fuse GNSS and INS position estimates.
        
        Parameters
        ----------
        gnss_position : np.ndarray or None
            GNSS position [x, y] in meters (None if unavailable).
        ins_position : np.ndarray
            INS dead reckoning position [x, y] in meters.
        gnss_available : bool
            Whether GNSS fix is available.
        gnss_anomaly : bool
            Whether GNSS anomaly is detected.
        disturbance_detected : bool
            Whether road disturbance is detected.
        disturbance_severity : float
            Severity of disturbance [0, 1].
        hdop : float
            GNSS HDOP value.
        integration_time : float
            Time since last GNSS update (seconds).
        dt : float
            Time step (seconds).
        
        Returns
        -------
        result : dict
            Fused position, weights, confidence, mode.
        """
        # Update confidence values
        self.confidence.update_gnss_confidence(
            gnss_available, gnss_anomaly, hdop, dt
        )
        self.confidence.update_imu_confidence(
            disturbance_detected, disturbance_severity, integration_time, dt
        )
        self.confidence.record()
        
        # Compute fusion weights from confidence
        gnss_conf = self.confidence.gnss_confidence
        imu_conf = self.confidence.imu_confidence
        
        if gnss_available and gnss_position is not None and gnss_conf > 0.05:
            # Both available: weighted fusion
            total = gnss_conf + imu_conf
            if total > 0:
                w_gnss = gnss_conf / total
                w_ins = imu_conf / total
            else:
                w_gnss = 0.5
                w_ins = 0.5
            
            fused_position = w_gnss * gnss_position + w_ins * ins_position
        else:
            # GNSS unavailable: INS only
            w_gnss = 0.0
            w_ins = 1.0
            fused_position = ins_position.copy()
        
        # Generate events
        mode = self.confidence.get_navigation_mode()
        
        return {
            'position': fused_position,
            'w_gnss': w_gnss,
            'w_ins': w_ins,
            'gnss_confidence': gnss_conf,
            'imu_confidence': imu_conf,
            'overall_confidence': self.confidence.compute_overall(),
            'navigation_mode': mode,
        }
    
    def fuse_velocity(self, gnss_speed: Optional[float],
                      ins_speed: float,
                      gnss_heading: Optional[float],
                      ins_heading: float) -> Dict:
        """
        Fuse velocity estimates.
        """
        gnss_conf = self.confidence.gnss_confidence
        imu_conf = self.confidence.imu_confidence
        
        if gnss_speed is not None and gnss_conf > 0.1:
            total = gnss_conf + imu_conf
            w_gnss = gnss_conf / total
            w_ins = imu_conf / total
            
            fused_speed = w_gnss * gnss_speed + w_ins * ins_speed
            
            # Heading fusion (circular mean)
            if gnss_heading is not None:
                fused_heading = np.arctan2(
                    w_gnss * np.sin(gnss_heading) + w_ins * np.sin(ins_heading),
                    w_gnss * np.cos(gnss_heading) + w_ins * np.cos(ins_heading)
                )
            else:
                fused_heading = ins_heading
        else:
            fused_speed = ins_speed
            fused_heading = ins_heading
        
        return {
            'speed': fused_speed,
            'heading': fused_heading,
        }
    
    def get_events(self, timestamp: float) -> List[str]:
        """Generate event messages based on current state."""
        events = []
        status = self.confidence.get_status()
        
        if status['gnss_confidence'] > 0.7:
            events.append("✅ GNSS healthy")
        elif status['gnss_confidence'] < 0.1:
            events.append("🔴 GNSS OUTAGE")
        elif status['gnss_anomaly']:
            events.append("⚠️ GNSS ANOMALY DETECTED")
        else:
            events.append("⚠️ GNSS DEGRADED")
        
        if status['imu_disturbance']:
            events.append("⚠️ ROAD DISTURBANCE DETECTED")
        else:
            events.append("✅ IMU normal")
        
        events.append(f"📍 Mode: {status['navigation_mode']}")
        
        return events
