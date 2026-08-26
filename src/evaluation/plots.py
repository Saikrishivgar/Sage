"""
Evaluation Plots Module
========================
Trajectory comparison, error vs time, confidence vs time,
acceleration magnitude, GNSS outage shading.

All charts use a clean light theme to match the SAGE UI.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Optional, List

# Consistent color palette for light backgrounds
COLORS = {
    'ground_truth': '#5f6368',
    'gnss': '#fbbc04',
    'baseline': '#ea4335',
    'adaptive': '#34a853',
    'fusion': '#1a73e8',
    'overall': '#1a73e8',
    'imu_conf': '#34a853',
    'gnss_conf': '#fbbc04',
}

LIGHT_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, sans-serif", size=12, color="#3c4043"),
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff",
    margin=dict(l=40, r=20, t=50, b=40),
)


def create_trajectory_plot(trajectories: Dict[str, Dict],
                           blackout_start: float = None,
                           blackout_end: float = None,
                           title: str = "Trajectory Comparison") -> go.Figure:
    """Create trajectory comparison plot."""
    fig = go.Figure()

    for name, traj in trajectories.items():
        fig.add_trace(go.Scatter(
            x=traj['x'], y=traj['y'],
            mode='lines',
            name=name,
            line=dict(
                color=traj.get('color', '#1a73e8'),
                dash=traj.get('dash', 'solid'),
                width=traj.get('width', 2),
            ),
            opacity=traj.get('opacity', 1.0),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="East (m)",
        yaxis_title="North (m)",
        height=450,
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#e5e7eb", borderwidth=1),
        **LIGHT_LAYOUT,
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    return fig


def create_error_plot(timestamps: np.ndarray,
                      errors: Dict[str, np.ndarray],
                      blackout_start: float = None,
                      blackout_end: float = None,
                      title: str = "Position Error Over Time") -> go.Figure:
    """Create position error over time plot with GNSS outage shading."""
    fig = go.Figure()

    colors = {
        'Baseline INS': COLORS['baseline'],
        'Adaptive': COLORS['adaptive'],
        'SAGE': COLORS['adaptive'],
        'AI+Fusion': COLORS['fusion'],
    }

    for name, err in errors.items():
        t = timestamps[:len(err)]
        fig.add_trace(go.Scatter(
            x=t, y=err,
            mode='lines',
            name=name,
            line=dict(color=colors.get(name, COLORS['fusion']), width=2),
        ))

    if blackout_start is not None and blackout_end is not None:
        fig.add_vrect(
            x0=blackout_start, x1=blackout_end,
            fillcolor="#ea4335", opacity=0.08,
            annotation_text="GNSS OUTAGE",
            annotation_position="top left",
            annotation_font=dict(color="#c5221f", size=11),
            line_width=0,
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Position Error (m)",
        height=350,
        showlegend=True,
        **LIGHT_LAYOUT,
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
        line=dict(color=COLORS['gnss_conf'], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=timestamps, y=imu_conf,
        mode='lines', name='IMU Confidence',
        line=dict(color=COLORS['imu_conf'], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=timestamps, y=overall_conf,
        mode='lines', name='Overall Confidence',
        line=dict(color=COLORS['overall'], width=2, dash='dot'),
    ))

    if blackout_start is not None and blackout_end is not None:
        fig.add_vrect(
            x0=blackout_start, x1=blackout_end,
            fillcolor="#ea4335", opacity=0.08,
            annotation_text="GNSS OUTAGE",
            annotation_position="top left",
            annotation_font=dict(color="#c5221f", size=11),
            line_width=0,
        )

    fig.update_layout(
        title="Sensor Confidence Over Time",
        xaxis_title="Time (s)",
        yaxis_title="Confidence [0–1]",
        yaxis_range=[-0.05, 1.1],
        height=300,
        showlegend=True,
        **LIGHT_LAYOUT,
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
            line=dict(color='rgba(234,67,53,0.25)', width=1),
        ))

    fig.add_trace(go.Scatter(
        x=timestamps, y=acc_mag,
        mode='lines', name='Processed',
        line=dict(color=COLORS['adaptive'], width=2),
    ))

    fig.add_hline(y=9.81, line_dash="dash", line_color="#80868b",
                  annotation_text="Gravity (9.81 m/s²)",
                  annotation_font=dict(color="#80868b", size=11))

    if blackout_start is not None and blackout_end is not None:
        fig.add_vrect(
            x0=blackout_start, x1=blackout_end,
            fillcolor="#ea4335", opacity=0.06,
            line_width=0,
        )

    fig.update_layout(
        title="Acceleration Magnitude",
        xaxis_title="Time (s)",
        yaxis_title="Acceleration (m/s²)",
        height=300,
        **LIGHT_LAYOUT,
    )

    return fig


def create_speed_plot(timestamps: np.ndarray,
                      speeds: Dict[str, np.ndarray],
                      blackout_start: float = None,
                      blackout_end: float = None) -> go.Figure:
    """Create speed comparison plot."""
    fig = go.Figure()

    colors = {
        'Ground Truth': COLORS['ground_truth'],
        'GNSS': COLORS['gnss'],
        'Baseline INS': COLORS['baseline'],
        'Adaptive': COLORS['adaptive'],
        'SAGE': COLORS['adaptive'],
    }

    for name, spd in speeds.items():
        t = timestamps[:len(spd)]
        fig.add_trace(go.Scatter(
            x=t, y=spd,
            mode='lines', name=name,
            line=dict(color=colors.get(name, COLORS['fusion']), width=2),
        ))

    if blackout_start is not None and blackout_end is not None:
        fig.add_vrect(
            x0=blackout_start, x1=blackout_end,
            fillcolor="#ea4335", opacity=0.06,
            line_width=0,
        )

    fig.update_layout(
        title="Speed Comparison",
        xaxis_title="Time (s)",
        yaxis_title="Speed (m/s)",
        height=300,
        **LIGHT_LAYOUT,
    )

    return fig


def create_comparison_table(results: List[Dict]) -> go.Figure:
    """Create method comparison table with light theme."""
    methods = [r['method'] for r in results]

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Method', 'Mean Error (m)', 'RMSE (m)', 'Max Error (m)', 'Drift %', 'Status'],
            fill_color='#f8f9fa',
            font=dict(color='#202124', size=13, family="Inter, sans-serif"),
            align='center',
            line_color='#e5e7eb',
            height=36,
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
                ['#ffffff'] * len(results),
                ['#ffffff'] * len(results),
                ['#ffffff'] * len(results),
                ['#ffffff'] * len(results),
                ['#ffffff'] * len(results),
                [('#e6f4ea' if r['pass_fail'] == 'PASS' else '#fce8e6') for r in results],
            ],
            font=dict(color='#3c4043', size=12, family="Inter, sans-serif"),
            align='center',
            line_color='#e5e7eb',
            height=32,
        ),
    )])

    fig.update_layout(
        title="SAGE vs Baseline",
        height=180,
        margin=dict(l=10, r=10, t=40, b=10),
        **{k: v for k, v in LIGHT_LAYOUT.items() if k != 'margin'},
    )

    return fig
