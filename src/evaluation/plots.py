"""
Evaluation Plots Module
========================
Trajectory comparison, error vs time, confidence vs time,
acceleration magnitude, GNSS outage shading.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Optional, List


def create_trajectory_plot(trajectories: Dict[str, Dict],
                           blackout_start: float = None,
                           blackout_end: float = None,
                           title: str = "Trajectory Comparison") -> go.Figure:
    """
    Create trajectory comparison plot.
    
    Parameters
    ----------
    trajectories : dict
        {name: {'x': array, 'y': array, 'color': str, 'dash': str}}
    """
    fig = go.Figure()
    
    for name, traj in trajectories.items():
        fig.add_trace(go.Scatter(
            x=traj['x'], y=traj['y'],
            mode='lines',
            name=name,
            line=dict(
                color=traj.get('color', 'blue'),
                dash=traj.get('dash', 'solid'),
                width=traj.get('width', 2),
            ),
            opacity=traj.get('opacity', 1.0),
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="East (m)",
        yaxis_title="North (m)",
        template="plotly_dark",
        height=500,
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    
    return fig


def create_error_plot(timestamps: np.ndarray,
                      errors: Dict[str, np.ndarray],
                      blackout_start: float = None,
                      blackout_end: float = None,
                      title: str = "Position Error vs Time") -> go.Figure:
    """Create position error over time plot with GNSS outage shading."""
    fig = go.Figure()
    
    colors = {'Baseline INS': '#ff6b6b', 'Adaptive': '#51cf66', 'AI+Fusion': '#339af0'}
    
    for name, err in errors.items():
        t = timestamps[:len(err)]
        fig.add_trace(go.Scatter(
            x=t, y=err,
            mode='lines',
            name=name,
            line=dict(color=colors.get(name, 'white'), width=2),
        ))
    
    # GNSS outage shading
    if blackout_start is not None and blackout_end is not None:
        fig.add_vrect(
            x0=blackout_start, x1=blackout_end,
            fillcolor="red", opacity=0.15,
            annotation_text="GNSS OUTAGE",
            annotation_position="top left",
            line_width=0,
        )
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Position Error (m)",
        template="plotly_dark",
        height=350,
        showlegend=True,
    )
    
    return fig


def create_confidence_plot(timestamps: np.ndarray,
                           gnss_conf: np.ndarray,
                           imu_conf: np.ndarray,
                           overall_conf: np.ndarray,
                           blackout_start: float = None,
                           blackout_end: float = None) -> go.Figure:
    """Create sensor confidence over time plot."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timestamps, y=gnss_conf,
        mode='lines', name='GNSS Confidence',
        line=dict(color='#ffd43b', width=2),
    ))
    fig.add_trace(go.Scatter(
        x=timestamps, y=imu_conf,
        mode='lines', name='IMU Confidence',
        line=dict(color='#51cf66', width=2),
    ))
    fig.add_trace(go.Scatter(
        x=timestamps, y=overall_conf,
        mode='lines', name='Overall Confidence',
        line=dict(color='#339af0', width=2, dash='dot'),
    ))
    
    if blackout_start is not None and blackout_end is not None:
        fig.add_vrect(
            x0=blackout_start, x1=blackout_end,
            fillcolor="red", opacity=0.15,
            annotation_text="GNSS OUTAGE",
            annotation_position="top left",
            line_width=0,
        )
    
    fig.update_layout(
        title="Sensor Confidence Over Time",
        xaxis_title="Time (s)",
        yaxis_title="Confidence [0-1]",
        yaxis_range=[-0.05, 1.1],
        template="plotly_dark",
        height=300,
        showlegend=True,
    )
    
    return fig


def create_acceleration_plot(timestamps: np.ndarray,
                             acc_mag: np.ndarray,
                             acc_mag_raw: np.ndarray = None,
                             blackout_start: float = None,
                             blackout_end: float = None) -> go.Figure:
    """Create acceleration magnitude plot."""
    fig = go.Figure()
    
    if acc_mag_raw is not None:
        fig.add_trace(go.Scatter(
            x=timestamps, y=acc_mag_raw,
            mode='lines', name='Raw',
            line=dict(color='rgba(255,107,107,0.3)', width=1),
        ))
    
    fig.add_trace(go.Scatter(
        x=timestamps, y=acc_mag,
        mode='lines', name='Processed',
        line=dict(color='#51cf66', width=2),
    ))
    
    # Gravity reference line
    fig.add_hline(y=9.81, line_dash="dash", line_color="gray",
                  annotation_text="Gravity (9.81 m/s²)")
    
    if blackout_start is not None and blackout_end is not None:
        fig.add_vrect(
            x0=blackout_start, x1=blackout_end,
            fillcolor="red", opacity=0.1,
            line_width=0,
        )
    
    fig.update_layout(
        title="Acceleration Magnitude",
        xaxis_title="Time (s)",
        yaxis_title="Acceleration (m/s²)",
        template="plotly_dark",
        height=300,
    )
    
    return fig


def create_speed_plot(timestamps: np.ndarray,
                      speeds: Dict[str, np.ndarray],
                      blackout_start: float = None,
                      blackout_end: float = None) -> go.Figure:
    """Create speed comparison plot."""
    fig = go.Figure()
    
    colors = {'Ground Truth': '#868e96', 'GNSS': '#ffd43b',
              'Baseline INS': '#ff6b6b', 'Adaptive': '#51cf66'}
    
    for name, spd in speeds.items():
        t = timestamps[:len(spd)]
        fig.add_trace(go.Scatter(
            x=t, y=spd,
            mode='lines', name=name,
            line=dict(color=colors.get(name, 'white'), width=2),
        ))
    
    if blackout_start is not None and blackout_end is not None:
        fig.add_vrect(
            x0=blackout_start, x1=blackout_end,
            fillcolor="red", opacity=0.1,
            line_width=0,
        )
    
    fig.update_layout(
        title="Speed Comparison",
        xaxis_title="Time (s)",
        yaxis_title="Speed (m/s)",
        template="plotly_dark",
        height=300,
    )
    
    return fig


def create_comparison_table(results: List[Dict]) -> go.Figure:
    """Create method comparison table."""
    methods = [r['method'] for r in results]
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Method', 'Mean Error (m)', 'RMSE (m)', 'Max Error (m)', 'Drift %', 'Status'],
            fill_color='#2c2c2c',
            font=dict(color='white', size=13),
            align='center',
        ),
        cells=dict(
            values=[
                methods,
                [f"{r['mean_error_m']:.2f}" for r in results],
                [f"{r['rmse_m']:.2f}" for r in results],
                [f"{r['max_error_m']:.2f}" for r in results],
                [f"{r['drift_percent']:.2f}" for r in results],
                [r['pass_fail'] for r in results],
            ],
            fill_color=[
                ['#1a1a2e'] * len(results),
                ['#1a1a2e'] * len(results),
                ['#1a1a2e'] * len(results),
                ['#1a1a2e'] * len(results),
                ['#1a1a2e'] * len(results),
                [('#2d5a2d' if r['pass_fail'] == 'PASS' else '#5a2d2d') for r in results],
            ],
            font=dict(color='white', size=12),
            align='center',
        ),
    )])
    
    fig.update_layout(
        title="Method Comparison",
        template="plotly_dark",
        height=200,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    
    return fig
