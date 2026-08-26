"""
Dead Reckoning Module
======================
Core dead reckoning engine: IMU → Position.

Pipeline:
    IMU → Calibration → Orientation → Coordinate Transform
    → Gravity Compensation → Acceleration → Velocity → Position

This is the BASELINE — shows natural drift problem.
"""

import numpy as np
from typing import Dict, Tuple, Optional


class DeadReckoningEngine:
    """
    Basic dead reckoning using IMU integration.
    
    Integrates accelerometer data to get velocity, then position.
    Uses trapezoidal integration for improved accuracy.
    
    This is intentionally a transparent baseline to demonstrate
    the drift problem that adaptive fusion solves.
    """
    
    def __init__(self, initial_position: np.ndarray = None,
                 initial_velocity: np.ndarray = None,
                 initial_heading: float = 0.0):
        """
        Parameters
        ----------
        initial_position : np.ndarray, shape (2,)
            Initial [x, y] position in meters (local frame).
        initial_velocity : np.ndarray, shape (2,)
            Initial [vx, vy] velocity in m/s.
        initial_heading : float
            Initial heading in radians (0=North, pi/2=East).
        """
        self.position = initial_position if initial_position is not None else np.zeros(2)
        self.velocity = initial_velocity if initial_velocity is not None else np.zeros(2)
        self.heading = initial_heading
        
        # History
        self.positions = [self.position.copy()]
        self.velocities = [self.velocity.copy()]
        self.headings = [self.heading]
        self.speeds = [np.linalg.norm(self.velocity)]
    
    def update(self, acc_forward: float, acc_lateral: float,
               gyro_z: float, dt: float,
               speed_constraint: Optional[float] = None) -> Dict:
        """
        Single dead reckoning step.
        
        Parameters
        ----------
        acc_forward : float
            Forward acceleration (m/s²), gravity-compensated.
        acc_lateral : float
            Lateral acceleration (m/s²), gravity-compensated.
        gyro_z : float
            Yaw rate (rad/s).
        dt : float
            Time step (seconds).
        speed_constraint : float, optional
            If provided, clamp speed to this value (from GNSS or ZUPT).
        
        Returns
        -------
        state : dict
            Current position, velocity, heading, speed.
        """
        # Update heading using gyroscope
        self.heading += gyro_z * dt
        self.heading = self.heading % (2 * np.pi)
        
        # Trapezoidal integration for velocity
        # Convert body-frame acceleration to navigation frame
        sin_h = np.sin(self.heading)
        cos_h = np.cos(self.heading)
        
        # Navigation frame acceleration (East, North)
        acc_east = acc_forward * sin_h + acc_lateral * cos_h
        acc_north = acc_forward * cos_h - acc_lateral * sin_h
        
        # Velocity integration (trapezoidal with previous)
        new_vx = self.velocity[0] + acc_east * dt
        new_vy = self.velocity[1] + acc_north * dt
        
        # Apply speed constraint if available
        speed = np.sqrt(new_vx**2 + new_vy**2)
        if speed_constraint is not None and speed > 0:
            scale = min(speed_constraint / speed, 1.0) if speed > speed_constraint else 1.0
            new_vx *= scale
            new_vy *= scale
            speed = np.sqrt(new_vx**2 + new_vy**2)
        
        # Non-negative speed constraint (vehicles don't fly backward easily)
        # Allow small negative for reversing
        if speed > 50.0:  # 180 km/h sanity check
            scale = 50.0 / speed
            new_vx *= scale
            new_vy *= scale
            speed = 50.0
        
        # Position integration (trapezoidal)
        new_x = self.position[0] + 0.5 * (self.velocity[0] + new_vx) * dt
        new_y = self.position[1] + 0.5 * (self.velocity[1] + new_vy) * dt
        
        # Update state
        self.velocity = np.array([new_vx, new_vy])
        self.position = np.array([new_x, new_y])
        
        # Store history
        self.positions.append(self.position.copy())
        self.velocities.append(self.velocity.copy())
        self.headings.append(self.heading)
        self.speeds.append(speed)
        
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'heading': self.heading,
            'speed': speed,
        }
    
    def get_trajectory(self) -> Dict:
        """Return complete trajectory history."""
        positions = np.array(self.positions)
        return {
            'x': positions[:, 0],
            'y': positions[:, 1],
            'headings': np.array(self.headings),
            'speeds': np.array(self.speeds),
        }
    
    def reset(self, position: np.ndarray = None,
              velocity: np.ndarray = None,
              heading: float = None):
        """Reset state (e.g., after GNSS correction)."""
        if position is not None:
            self.position = position.copy()
        if velocity is not None:
            self.velocity = velocity.copy()
        if heading is not None:
            self.heading = heading


def run_dead_reckoning(acc_forward: np.ndarray, acc_lateral: np.ndarray,
                       gyro_z: np.ndarray, dt_array: np.ndarray,
                       initial_position: np.ndarray = None,
                       initial_velocity: np.ndarray = None,
                       initial_heading: float = 0.0) -> Dict:
    """
    Run dead reckoning over entire dataset.
    
    Parameters
    ----------
    acc_forward : np.ndarray
        Forward acceleration array (m/s²).
    acc_lateral : np.ndarray
        Lateral acceleration array (m/s²).
    gyro_z : np.ndarray
        Yaw rate array (rad/s).
    dt_array : np.ndarray
        Time step array (seconds).
    
    Returns
    -------
    results : dict
        Trajectory with x, y, headings, speeds arrays.
    """
    engine = DeadReckoningEngine(
        initial_position=initial_position,
        initial_velocity=initial_velocity,
        initial_heading=initial_heading,
    )
    
    n = len(acc_forward)
    for i in range(n):
        engine.update(acc_forward[i], acc_lateral[i], gyro_z[i], dt_array[i])
    
    return engine.get_trajectory()
