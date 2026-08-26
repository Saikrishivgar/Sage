"""
GNSS Anomaly Detection Module
===============================
Detects unreliable GNSS measurements EVEN when GNSS is available.

Detects:
- Sudden position jumps
- Unrealistic speed
- Large disagreement with INS estimate
- HDOP degradation

Output: GNSS confidence ∈ [0, 1]
"""

import numpy as np
from typing import Dict, Optional, Tuple


class GNSSAnomalyDetector:
    """
    Rule-based GNSS anomaly detector.
    
    Prototype — planned to be replaced by ML-based GNSS quality
    estimation in Phase 2.
    """
    
    def __init__(self):
        self.prev_gnss_position = None
        self.prev_gnss_time = None
        self.prev_gnss_speed = None
        self.anomaly_count = 0
        self.history = []
    
    def detect(self, gnss_lat: float, gnss_lon: float,
               gnss_speed: float, gnss_hdop: float,
               ins_x: float, ins_y: float,
               timestamp: float,
               lat0: float, lon0: float) -> Dict:
        """
        Check GNSS measurement for anomalies.
        
        Parameters
        ----------
        gnss_lat, gnss_lon : float
            GNSS position (degrees).
        gnss_speed : float
            GNSS-reported speed (m/s).
        gnss_hdop : float
            Horizontal Dilution of Precision.
        ins_x, ins_y : float
            Current INS position estimate (meters, local frame).
        timestamp : float
            Current timestamp (seconds).
        lat0, lon0 : float
            Origin coordinates for local frame conversion.
        
        Returns
        -------
        result : dict
            anomaly_detected, anomaly_type, gnss_confidence, details.
        """
        anomalies = []
        confidence = 1.0
        
        # Convert GNSS to local frame
        m_per_deg_lat = 111132.92
        m_per_deg_lon = 111132.92 * np.cos(np.radians(lat0))
        gnss_x = (gnss_lon - lon0) * m_per_deg_lon
        gnss_y = (gnss_lat - lat0) * m_per_deg_lat
        
        # Check 1: Position jump
        if self.prev_gnss_position is not None and self.prev_gnss_time is not None:
            dt = timestamp - self.prev_gnss_time
            if dt > 0:
                dx = gnss_x - self.prev_gnss_position[0]
                dy = gnss_y - self.prev_gnss_position[1]
                jump_distance = np.sqrt(dx**2 + dy**2)
                implied_speed = jump_distance / dt
                
                # Position jump > 50m in one step
                if jump_distance > 50.0:
                    anomalies.append('POSITION_JUMP')
                    confidence *= 0.2
                
                # Implied speed > 200 km/h
                if implied_speed > 55.0:  # ~200 km/h
                    anomalies.append('UNREALISTIC_SPEED')
                    confidence *= 0.3
        
        # Check 2: GNSS speed anomaly
        if gnss_speed > 50.0:  # > 180 km/h
            anomalies.append('SPEED_TOO_HIGH')
            confidence *= 0.3
        
        # Check 3: INS disagreement
        ins_gnss_distance = np.sqrt((gnss_x - ins_x)**2 + (gnss_y - ins_y)**2)
        if ins_gnss_distance > 30.0:  # > 30m disagreement
            anomalies.append('INS_DISAGREEMENT')
            confidence *= 0.5
        
        # Check 4: HDOP degradation
        if gnss_hdop > 3.0:
            anomalies.append('HIGH_HDOP')
            confidence *= max(0.3, 1.0 - (gnss_hdop - 1.0) * 0.15)
        elif gnss_hdop > 2.0:
            confidence *= 0.8
        
        anomaly_detected = len(anomalies) > 0
        if anomaly_detected:
            self.anomaly_count += 1
        
        # Store previous
        self.prev_gnss_position = np.array([gnss_x, gnss_y])
        self.prev_gnss_time = timestamp
        self.prev_gnss_speed = gnss_speed
        
        result = {
            'anomaly_detected': anomaly_detected,
            'anomaly_types': anomalies,
            'gnss_confidence': float(np.clip(confidence, 0.0, 1.0)),
            'gnss_position_local': np.array([gnss_x, gnss_y]),
            'ins_gnss_distance': ins_gnss_distance,
            'anomaly_count': self.anomaly_count,
        }
        
        self.history.append(result)
        return result
    
    def simulate_anomaly(self, gnss_x: float, gnss_y: float,
                         anomaly_type: str = 'jump') -> Tuple[float, float]:
        """
        Simulate a GNSS anomaly for stress testing.
        
        Parameters
        ----------
        gnss_x, gnss_y : float
            Original GNSS position.
        anomaly_type : str
            'jump': sudden 50m position shift
            'drift': gradual position drift
        
        Returns
        -------
        anomalous_x, anomalous_y : float
            Modified GNSS position.
        """
        if anomaly_type == 'jump':
            offset = np.random.uniform(30, 80)
            angle = np.random.uniform(0, 2 * np.pi)
            return gnss_x + offset * np.cos(angle), gnss_y + offset * np.sin(angle)
        elif anomaly_type == 'drift':
            return gnss_x + np.random.normal(0, 10), gnss_y + np.random.normal(0, 10)
        return gnss_x, gnss_y
