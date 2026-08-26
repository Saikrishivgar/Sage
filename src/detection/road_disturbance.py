"""
Road / Motion Disturbance Detector
====================================
Lightweight rule-based detector for road surface disturbances.

Uses acceleration magnitude to detect abnormal short-duration spikes.

Classification:
- NORMAL: no disturbance
- DISTURBANCE: generic disturbance detected
- ROUGH_ROAD: sustained elevated vibration
- POTHOLE_LIKE: sharp single spike
- HARD_BRAKING: sustained forward deceleration

IMPORTANT: This is a RULE-BASED detector.
Not an AI pothole model. No ML model has been trained.
"""

import numpy as np
from typing import Dict, List, Optional


class RoadDisturbanceDetector:
    """
    Detects road surface disturbances from acceleration data.
    
    Rule-based prototype — planned ML replacement in Phase 2.
    """
    
    def __init__(self, fs: float = 100.0):
        """
        Parameters
        ----------
        fs : float
            Sampling frequency (Hz).
        """
        self.fs = fs
        self.gravity = 9.81
        
        # Detection thresholds (documented)
        self.spike_threshold = 3.0   # m/s² above/below gravity for spike
        self.rough_threshold = 1.5   # m/s² sustained deviation
        self.braking_threshold = -3.0  # m/s² forward deceleration
        
        # Window for sustained detection
        self.window_size = int(0.5 * fs)  # 500ms window
        
        # History
        self.acc_mag_buffer = []
        self.acc_forward_buffer = []
        self.detections: List[Dict] = []
        self.history = []
    
    def detect(self, acc_x: float, acc_y: float, acc_z: float,
               acc_forward: float = 0.0,
               timestamp: float = 0.0) -> Dict:
        """
        Detect disturbance from single accelerometer sample.
        
        Parameters
        ----------
        acc_x, acc_y, acc_z : float
            Raw accelerometer readings (m/s²).
        acc_forward : float
            Forward-axis acceleration (m/s²).
        timestamp : float
            Current timestamp.
        
        Returns
        -------
        result : dict
            detected, classification, severity, imu_confidence_factor.
        """
        # Acceleration magnitude
        acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        deviation = abs(acc_mag - self.gravity)
        
        # Update buffer
        self.acc_mag_buffer.append(deviation)
        self.acc_forward_buffer.append(acc_forward)
        
        if len(self.acc_mag_buffer) > self.window_size:
            self.acc_mag_buffer = self.acc_mag_buffer[-self.window_size:]
            self.acc_forward_buffer = self.acc_forward_buffer[-self.window_size:]
        
        # Classification
        classification = 'NORMAL'
        severity = 0.0
        detected = False
        
        # Check for spike (pothole-like)
        if deviation > self.spike_threshold:
            detected = True
            severity = min(1.0, deviation / 10.0)
            
            if deviation > 5.0:
                classification = 'POTHOLE_LIKE'
            else:
                classification = 'DISTURBANCE'
        
        # Check for sustained rough road
        if len(self.acc_mag_buffer) >= self.window_size // 2:
            window_std = np.std(self.acc_mag_buffer[-self.window_size // 2:])
            window_mean = np.mean(self.acc_mag_buffer[-self.window_size // 2:])
            
            if window_mean > self.rough_threshold and window_std > 0.5:
                detected = True
                severity = max(severity, min(1.0, window_mean / 5.0))
                classification = 'ROUGH_ROAD'
        
        # Check for hard braking
        if acc_forward < self.braking_threshold:
            detected = True
            severity = max(severity, min(1.0, abs(acc_forward) / 8.0))
            classification = 'HARD_BRAKING'
        
        # IMU confidence factor: how much to trust IMU during disturbance
        # 1.0 = full trust, 0.0 = no trust
        if detected:
            imu_confidence_factor = max(0.3, 1.0 - severity * 0.7)
        else:
            imu_confidence_factor = 1.0
        
        result = {
            'detected': detected,
            'classification': classification,
            'severity': severity,
            'acc_magnitude': acc_mag,
            'deviation': deviation,
            'imu_confidence_factor': imu_confidence_factor,
            'timestamp': timestamp,
        }
        
        if detected:
            self.detections.append(result)
        
        self.history.append(result)
        
        return result
    
    def get_summary(self) -> Dict:
        """Get detection summary."""
        if not self.detections:
            return {
                'total_detections': 0,
                'classifications': {},
                'max_severity': 0.0,
            }
        
        classifications = {}
        for d in self.detections:
            c = d['classification']
            classifications[c] = classifications.get(c, 0) + 1
        
        return {
            'total_detections': len(self.detections),
            'classifications': classifications,
            'max_severity': max(d['severity'] for d in self.detections),
            'mean_severity': np.mean([d['severity'] for d in self.detections]),
        }
