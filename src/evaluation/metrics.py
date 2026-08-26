"""
Accuracy Metrics Module
========================
Comprehensive accuracy evaluation for navigation estimates.

Calculates:
1. Instantaneous position error
2. Mean error
3. RMSE
4. Maximum error
5. Velocity MAE / RMSE
6. Distance travelled
7. GNSS-denied distance
8. Final blackout error
9. Drift percentage
10. Per-segment metrics (overall + blackout-only)
"""

import numpy as np
from typing import Dict, Optional


def position_error(estimated_x: np.ndarray, estimated_y: np.ndarray,
                   truth_x: np.ndarray, truth_y: np.ndarray) -> np.ndarray:
    """
    Instantaneous position error (meters).
    
    e_t = ||p_estimated,t - p_groundtruth,t||
    """
    return np.sqrt((estimated_x - truth_x)**2 + (estimated_y - truth_y)**2)


def mean_error(errors: np.ndarray) -> float:
    """
    Mean position error.
    
    mean_error = mean(e)
    """
    return float(np.mean(errors))


def rmse(errors: np.ndarray) -> float:
    """
    Root Mean Square Error.
    
    RMSE = sqrt(mean(e²))
    """
    return float(np.sqrt(np.mean(errors**2)))


def max_error(errors: np.ndarray) -> float:
    """Maximum position error."""
    return float(np.max(errors))


def velocity_mae(estimated_speed: np.ndarray,
                 truth_speed: np.ndarray) -> float:
    """Mean Absolute Error of velocity."""
    return float(np.mean(np.abs(estimated_speed - truth_speed)))


def velocity_rmse(estimated_speed: np.ndarray,
                  truth_speed: np.ndarray) -> float:
    """RMSE of velocity."""
    return float(np.sqrt(np.mean((estimated_speed - truth_speed)**2)))


def distance_travelled(x: np.ndarray, y: np.ndarray) -> float:
    """
    Total distance travelled along trajectory.
    
    Sum of inter-point distances.
    """
    dx = np.diff(x)
    dy = np.diff(y)
    return float(np.sum(np.sqrt(dx**2 + dy**2)))


def drift_percentage(final_error: float, total_distance: float) -> float:
    """
    Drift as percentage of distance travelled.
    
    drift_percent = (position_error / distance_travelled) * 100
    """
    if total_distance <= 0:
        return 0.0
    return (final_error / total_distance) * 100.0


def compute_full_metrics(estimated_x: np.ndarray, estimated_y: np.ndarray,
                         truth_x: np.ndarray, truth_y: np.ndarray,
                         estimated_speed: np.ndarray = None,
                         truth_speed: np.ndarray = None,
                         blackout_mask: np.ndarray = None,
                         target_drift: float = 10.0) -> Dict:
    """
    Compute comprehensive accuracy metrics.
    
    Parameters
    ----------
    estimated_x, estimated_y : np.ndarray
        Estimated trajectory (meters).
    truth_x, truth_y : np.ndarray
        Ground truth trajectory (meters).
    estimated_speed, truth_speed : np.ndarray, optional
        Speed estimates and ground truth.
    blackout_mask : np.ndarray, optional
        Boolean mask where True = GNSS denied.
    target_drift : float
        Target drift percentage for pass/fail.
    
    Returns
    -------
    metrics : dict
        All computed metrics.
    """
    # Ensure same length
    n = min(len(estimated_x), len(truth_x))
    est_x = estimated_x[:n]
    est_y = estimated_y[:n]
    gt_x = truth_x[:n]
    gt_y = truth_y[:n]
    
    # Position errors
    errors = position_error(est_x, est_y, gt_x, gt_y)
    
    # Distance
    dist_truth = distance_travelled(gt_x, gt_y)
    dist_est = distance_travelled(est_x, est_y)
    
    # Final error
    final_err = float(errors[-1]) if len(errors) > 0 else 0.0
    
    # Drift
    drift = drift_percentage(final_err, dist_truth)
    
    metrics = {
        'mean_error_m': mean_error(errors),
        'rmse_m': rmse(errors),
        'max_error_m': max_error(errors),
        'final_error_m': final_err,
        'distance_travelled_m': dist_truth,
        'estimated_distance_m': dist_est,
        'drift_percent': drift,
        'target_drift_percent': target_drift,
        'pass_fail': 'PASS' if drift < target_drift else 'FAIL',
        'position_errors': errors,
    }
    
    # Velocity metrics
    if estimated_speed is not None and truth_speed is not None:
        n_speed = min(len(estimated_speed), len(truth_speed))
        metrics['velocity_mae_ms'] = velocity_mae(
            estimated_speed[:n_speed], truth_speed[:n_speed]
        )
        metrics['velocity_rmse_ms'] = velocity_rmse(
            estimated_speed[:n_speed], truth_speed[:n_speed]
        )
    
    # Blackout-specific metrics
    if blackout_mask is not None:
        mask = blackout_mask[:n]
        if np.any(mask):
            blackout_errors = errors[mask]
            blackout_gt_x = gt_x[mask]
            blackout_gt_y = gt_y[mask]
            
            blackout_dist = distance_travelled(blackout_gt_x, blackout_gt_y)
            blackout_final_err = float(blackout_errors[-1])
            
            metrics['blackout_mean_error_m'] = mean_error(blackout_errors)
            metrics['blackout_rmse_m'] = rmse(blackout_errors)
            metrics['blackout_max_error_m'] = max_error(blackout_errors)
            metrics['blackout_final_error_m'] = blackout_final_err
            metrics['blackout_distance_m'] = blackout_dist
            metrics['blackout_drift_percent'] = drift_percentage(
                blackout_final_err, blackout_dist
            )
            metrics['blackout_duration_samples'] = int(np.sum(mask))
    
    return metrics


def format_metrics_table(metrics: Dict, method_name: str = "Method") -> str:
    """Format metrics as a display string."""
    lines = [
        f"{'Metric':<25} {'Value':>12}",
        "-" * 40,
        f"{'Mean Error':<25} {metrics['mean_error_m']:>10.2f} m",
        f"{'RMSE':<25} {metrics['rmse_m']:>10.2f} m",
        f"{'Max Error':<25} {metrics['max_error_m']:>10.2f} m",
        f"{'Final Error':<25} {metrics['final_error_m']:>10.2f} m",
        f"{'Distance Travelled':<25} {metrics['distance_travelled_m']:>10.1f} m",
        f"{'Drift':<25} {metrics['drift_percent']:>10.2f} %",
        f"{'Target':<25} {'<' + str(metrics['target_drift_percent']) + '%':>12}",
        f"{'Status':<25} {metrics['pass_fail']:>12}",
    ]
    
    if 'blackout_rmse_m' in metrics:
        lines.extend([
            "",
            "--- GNSS Blackout Period ---",
            f"{'Blackout Mean Error':<25} {metrics['blackout_mean_error_m']:>10.2f} m",
            f"{'Blackout RMSE':<25} {metrics['blackout_rmse_m']:>10.2f} m",
            f"{'Blackout Max Error':<25} {metrics['blackout_max_error_m']:>10.2f} m",
            f"{'Blackout Drift':<25} {metrics['blackout_drift_percent']:>10.2f} %",
        ])
    
    return "\n".join(lines)
