"""
Intelligent Dead Reckoning System — Streamlit Dashboard
=========================================================
SIH Problem Statement 168: AI-ML based Intelligent Dead Reckoning
System for Seamless Navigation

SIMULATED DATA — NOT REAL EXPERIMENTAL RESULTS
Prototype confidence engine — planned ML replacement in Phase 2.

Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import load_dataset
from src.data.preprocessing import preprocess_imu, compute_dt
from src.navigation.dead_reckoning import DeadReckoningEngine
from src.navigation.fusion import AdaptiveFusion
from src.detection.gnss_anomaly import GNSSAnomalyDetector
from src.detection.road_disturbance import RoadDisturbanceDetector
from src.evaluation.metrics import compute_full_metrics, position_error
from src.evaluation.plots import (
    create_trajectory_plot, create_error_plot, create_confidence_plot,
    create_acceleration_plot, create_speed_plot, create_comparison_table
)
from src.sensors.coordinate_transform import latlon_to_local, local_to_latlon
from src.visualization.dashboard import (
    apply_mobile_style, render_header, render_mode_indicator,
    render_confidence_bar, render_accuracy_panel, render_events_panel,
    render_section_title, render_metric
)

st.set_page_config(
    page_title="Intelligent Dead Reckoning",
    page_icon="🧭",
    layout="centered",
    initial_sidebar_state="collapsed",
)

apply_mobile_style()


# ============================================================
# PIPELINE FUNCTIONS
# ============================================================

@st.cache_data
def load_and_preprocess():
    """Load dataset and run preprocessing."""
    filepath = os.path.join(os.path.dirname(__file__), 'data', 'sample', 'simulated_drive.csv')
    df, meta = load_dataset(filepath)
    df_processed, prep_info = preprocess_imu(df, meta)
    return df_processed, meta, prep_info


def run_baseline_dr(df, meta):
    """Run baseline dead reckoning (no fusion, no corrections)."""
    engine = DeadReckoningEngine(
        initial_position=np.array([df['gt_x'].iloc[0], df['gt_y'].iloc[0]]),
        initial_velocity=np.array([0.0, 0.0]),
        initial_heading=float(df['gt_heading'].iloc[0]),
    )
    
    dt_arr = compute_dt(df)
    
    for i in range(1, len(df)):
        # Use bias-corrected accelerometer (forward = acc_x after preprocessing)
        acc_fwd = df['acc_x'].iloc[i]
        acc_lat = df['acc_y'].iloc[i]
        gyro_z = df['gyro_z'].iloc[i]
        
        engine.update(acc_fwd, acc_lat, gyro_z, dt_arr[i])
    
    traj = engine.get_trajectory()
    return traj


def run_adaptive_pipeline(df, meta, blackout_start, blackout_end,
                          disturbance_enabled=True,
                          gnss_anomaly_time=None):
    """
    Run full adaptive fusion pipeline.
    
    Strategy:
    - When GNSS available: Use GNSS position directly with IMU heading smoothing
    - When GNSS denied: Fall back to dead reckoning from last known position
    - Adaptive confidence adjusts fusion weights in real-time
    
    Returns trajectory, confidence history, events, metrics.
    """
    lat0 = meta['lat0']
    lon0 = meta['lon0']
    
    # Initialize components
    fusion = AdaptiveFusion()
    gnss_detector = GNSSAnomalyDetector()
    road_detector = RoadDisturbanceDetector(fs=meta.get('imu_rate', 100))
    
    dt_arr = compute_dt(df)
    
    # Results storage
    fused_x = [df['gt_x'].iloc[0]]
    fused_y = [df['gt_y'].iloc[0]]
    fused_speeds = [0.0]
    navigation_modes = ["GNSS + INS"]
    blackout_mask = [False]
    
    # Current best estimate (starts at origin)
    current_pos = np.array([df['gt_x'].iloc[0], df['gt_y'].iloc[0]])
    current_heading = float(df['gt_heading'].iloc[0])
    current_speed = 0.0
    last_gnss_pos = current_pos.copy()
    last_gnss_heading = current_heading
    last_gnss_speed = 0.0
    
    gnss_outage_start_time = None
    
    # DR engine for pure INS tracking (used during outage)
    dr_velocity = np.array([0.0, 0.0])
    
    for i in range(1, len(df)):
        t = df['timestamp'].iloc[i]
        dt = dt_arr[i]
        
        # --- IMU data ---
        acc_fwd = df['acc_x'].iloc[i]
        acc_lat = df['acc_y'].iloc[i]
        gyro_z = df['gyro_z'].iloc[i]
        
        # Update heading from gyro
        current_heading += gyro_z * dt
        current_heading = current_heading % (2 * np.pi)
        
        # --- Determine GNSS availability ---
        gnss_available = not (blackout_start <= t <= blackout_end)
        is_blackout = (blackout_start <= t <= blackout_end)
        blackout_mask.append(is_blackout)
        
        # Track integration time during outage
        if not gnss_available:
            if gnss_outage_start_time is None:
                gnss_outage_start_time = t
                # Initialize DR velocity from last GNSS speed
                dr_velocity = np.array([
                    last_gnss_speed * np.sin(current_heading),
                    last_gnss_speed * np.cos(current_heading),
                ])
            integration_time = t - gnss_outage_start_time
        else:
            gnss_outage_start_time = None
            integration_time = 0.0
        
        # --- GNSS Processing ---
        gnss_position = None
        gnss_speed_val = None
        gnss_heading_val = None
        gnss_anomaly = False
        hdop = 1.0
        
        if gnss_available and not pd.isna(df['gnss_lat'].iloc[i]):
            gnss_lat = df['gnss_lat'].iloc[i]
            gnss_lon = df['gnss_lon'].iloc[i]
            gnss_speed_val = df['gnss_speed'].iloc[i]
            hdop = df['gnss_hdop'].iloc[i] if 'gnss_hdop' in df.columns else 1.0
            
            # Simulate GNSS anomaly if requested
            if gnss_anomaly_time is not None and abs(t - gnss_anomaly_time) < 2.0:
                gnss_lat += 0.0005  # ~55m jump
                gnss_lon += 0.0005
            
            # Convert GNSS to local frame
            gx, gy = latlon_to_local(
                np.array([gnss_lat]), np.array([gnss_lon]), lat0, lon0
            )
            gnss_position = np.array([gx[0], gy[0]])
            gnss_speed_val = float(gnss_speed_val)
            gnss_heading_val = float(df['gnss_heading'].iloc[i]) if 'gnss_heading' in df.columns else None
            
            # GNSS anomaly detection — compare against fused position, not drifted INS
            anomaly_result = gnss_detector.detect(
                gnss_lat, gnss_lon, gnss_speed_val, hdop,
                current_pos[0], current_pos[1], t, lat0, lon0
            )
            gnss_anomaly = anomaly_result['anomaly_detected']
            
            # Store last good GNSS
            if not gnss_anomaly:
                last_gnss_pos = gnss_position.copy()
                last_gnss_heading = gnss_heading_val if gnss_heading_val is not None else current_heading
                last_gnss_speed = gnss_speed_val
        
        # --- Dead Reckoning Step (for GNSS-denied mode) ---
        # Propagate using speed + heading (velocity-based DR)
        # During GNSS outage: integrate acceleration to update velocity
        if not gnss_available:
            # Update DR velocity with accelerometer
            sin_h = np.sin(current_heading)
            cos_h = np.cos(current_heading)
            acc_east = acc_fwd * sin_h + acc_lat * cos_h
            acc_north = acc_fwd * cos_h - acc_lat * sin_h
            dr_velocity[0] += acc_east * dt
            dr_velocity[1] += acc_north * dt
            
            # Clamp DR speed
            dr_speed = np.linalg.norm(dr_velocity)
            if dr_speed > 50.0:
                dr_velocity *= 50.0 / dr_speed
            
            ins_position = current_pos + dr_velocity * dt
            ins_speed = np.linalg.norm(dr_velocity)
        else:
            # When GNSS available: use speed + heading for INS prediction
            ins_speed = last_gnss_speed
            ins_position = current_pos + np.array([
                ins_speed * np.sin(current_heading),
                ins_speed * np.cos(current_heading),
            ]) * dt
        
        # --- Road Disturbance Detection ---
        disturbance_detected = False
        disturbance_severity = 0.0
        
        if disturbance_enabled:
            dist_result = road_detector.detect(
                df['acc_x_raw'].iloc[i] if 'acc_x_raw' in df.columns else df['acc_x'].iloc[i],
                df['acc_y_raw'].iloc[i] if 'acc_y_raw' in df.columns else df['acc_y'].iloc[i],
                df['acc_z_raw'].iloc[i] if 'acc_z_raw' in df.columns else df['acc_z'].iloc[i],
                acc_fwd, t
            )
            disturbance_detected = dist_result['detected']
            disturbance_severity = dist_result['severity']
        
        # --- Adaptive Fusion ---
        fusion_result = fusion.fuse_position(
            gnss_position=gnss_position,
            ins_position=ins_position,
            gnss_available=gnss_available,
            gnss_anomaly=gnss_anomaly,
            disturbance_detected=disturbance_detected,
            disturbance_severity=disturbance_severity,
            hdop=hdop,
            integration_time=integration_time,
            dt=dt,
        )
        
        fused_pos = fusion_result['position']
        current_pos = fused_pos.copy()
        current_speed = ins_speed
        
        fused_x.append(fused_pos[0])
        fused_y.append(fused_pos[1])
        fused_speeds.append(current_speed)
        navigation_modes.append(fusion_result['navigation_mode'])
    
    # Build results
    fused_x = np.array(fused_x)
    fused_y = np.array(fused_y)
    fused_speeds = np.array(fused_speeds)
    blackout_mask = np.array(blackout_mask)
    
    # Get confidence history
    conf_history = fusion.confidence.history
    
    # Events for display
    final_events = fusion.get_events(df['timestamp'].iloc[-1])
    
    return {
        'x': fused_x,
        'y': fused_y,
        'speeds': fused_speeds,
        'modes': navigation_modes,
        'blackout_mask': blackout_mask,
        'gnss_confidence': np.array(conf_history['gnss']),
        'imu_confidence': np.array(conf_history['imu']),
        'overall_confidence': np.array(conf_history['overall']),
        'events': final_events,
        'road_detections': road_detector.get_summary(),
    }


# ============================================================
# MAIN APP
# ============================================================

def main():
    render_header()
    
    # Load data
    try:
        df, meta, prep_info = load_and_preprocess()
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        st.info("Run `python generate_dataset.py` first to create the simulated dataset.")
        return
    
    timestamps = df['timestamp'].values
    duration = timestamps[-1] - timestamps[0]
    
    # ============================================================
    # CONTROLS
    # ============================================================
    render_section_title("🎮 CONTROLS")
    
    # Stress test scenarios
    col1, col2, col3 = st.columns(3)
    with col1:
        scenario_normal = st.button("✅ NORMAL", use_container_width=True)
    with col2:
        scenario_gnss_loss = st.button("📡 GNSS LOSS", use_container_width=True)
    with col3:
        scenario_rough = st.button("🛣️ ROUGH ROAD", use_container_width=True)
    
    col4, col5, col6 = st.columns(3)
    with col4:
        scenario_jump = st.button("⚡ GNSS JUMP", use_container_width=True)
    with col5:
        scenario_combo = st.button("💥 COMBINED", use_container_width=True)
    with col6:
        scenario_reset = st.button("🔄 RESET", use_container_width=True)
    
    # Determine scenario
    if 'scenario' not in st.session_state or scenario_reset:
        st.session_state.scenario = 'normal'
    
    if scenario_normal:
        st.session_state.scenario = 'normal'
    elif scenario_gnss_loss:
        st.session_state.scenario = 'gnss_loss'
    elif scenario_rough:
        st.session_state.scenario = 'rough_road'
    elif scenario_jump:
        st.session_state.scenario = 'gnss_jump'
    elif scenario_combo:
        st.session_state.scenario = 'combined'
    
    scenario = st.session_state.scenario
    
    # Scenario parameters
    if scenario == 'normal':
        blackout_start, blackout_end = 999, 999  # No blackout
        disturbance_enabled = False
        gnss_anomaly_time = None
        scenario_label = "NORMAL — Full GNSS + INS"
    elif scenario == 'gnss_loss':
        blackout_start, blackout_end = 30, 60
        disturbance_enabled = False
        gnss_anomaly_time = None
        scenario_label = "GNSS LOSS — 30s to 60s blackout"
    elif scenario == 'rough_road':
        blackout_start, blackout_end = 999, 999
        disturbance_enabled = True
        gnss_anomaly_time = None
        scenario_label = "ROUGH ROAD — Disturbance detection active"
    elif scenario == 'gnss_jump':
        blackout_start, blackout_end = 999, 999
        disturbance_enabled = False
        gnss_anomaly_time = 45.0
        scenario_label = "GNSS JUMP — Position anomaly at 45s"
    elif scenario == 'combined':
        blackout_start, blackout_end = 30, 60
        disturbance_enabled = True
        gnss_anomaly_time = None
        scenario_label = "COMBINED — GNSS Loss (30-60s) + Rough Road"
    else:
        blackout_start, blackout_end = 30, 60
        disturbance_enabled = True
        gnss_anomaly_time = None
        scenario_label = "Default"
    
    # Custom controls expander
    with st.expander("⚙️ Custom Scenario"):
        custom_start = st.slider("GNSS Outage Start (s)", 0, int(duration) - 5, int(blackout_start) if blackout_start < 900 else 30)
        custom_duration = st.slider("GNSS Outage Duration (s)", 5, 60, int(blackout_end - blackout_start) if blackout_start < 900 else 30)
        custom_disturbance = st.checkbox("Enable Road Disturbance Detection", value=disturbance_enabled)
        if st.button("Apply Custom"):
            blackout_start = custom_start
            blackout_end = custom_start + custom_duration
            disturbance_enabled = custom_disturbance
            scenario_label = f"CUSTOM — Blackout {blackout_start}s-{blackout_end}s"
    
    st.markdown(f"<div style='text-align:center; color:#6b7280; font-size:12px; margin:8px 0;'>📋 {scenario_label}</div>", unsafe_allow_html=True)
    
    # ============================================================
    # RUN PIPELINES
    # ============================================================
    
    # Baseline Dead Reckoning
    baseline_traj = run_baseline_dr(df, meta)
    
    # Adaptive Pipeline
    adaptive_result = run_adaptive_pipeline(
        df, meta, blackout_start, blackout_end,
        disturbance_enabled=disturbance_enabled,
        gnss_anomaly_time=gnss_anomaly_time,
    )
    
    # ============================================================
    # MAP
    # ============================================================
    render_section_title("🗺️ MAP")
    
    lat0 = meta['lat0']
    lon0 = meta['lon0']
    
    # Create folium map
    m = folium.Map(location=[lat0, lon0], zoom_start=16, tiles='OpenStreetMap')
    
    # Ground truth trajectory
    gt_lats = df['gt_lat'].values
    gt_lons = df['gt_lon'].values
    gt_coords = list(zip(gt_lats[::10], gt_lons[::10]))
    folium.PolyLine(gt_coords, color='#868e96', weight=3, opacity=0.7,
                    tooltip='Ground Truth').add_to(m)
    
    # GNSS trajectory (where available)
    gnss_mask = ~df['gnss_lat'].isna()
    if gnss_mask.any():
        gnss_lats = df.loc[gnss_mask, 'gnss_lat'].values
        gnss_lons = df.loc[gnss_mask, 'gnss_lon'].values
        gnss_coords = list(zip(gnss_lats[::5], gnss_lons[::5]))
        if len(gnss_coords) > 1:
            folium.PolyLine(gnss_coords, color='#ffd43b', weight=2, opacity=0.5,
                            dash_array='5', tooltip='GNSS').add_to(m)
    
    # Baseline INS
    baseline_lats, baseline_lons = local_to_latlon(
        baseline_traj['x'], baseline_traj['y'], lat0, lon0
    )
    baseline_coords = list(zip(baseline_lats[::10], baseline_lons[::10]))
    folium.PolyLine(baseline_coords, color='#ff6b6b', weight=2, opacity=0.6,
                    dash_array='8', tooltip='Baseline INS').add_to(m)
    
    # Adaptive result
    adapt_lats, adapt_lons = local_to_latlon(
        adaptive_result['x'], adaptive_result['y'], lat0, lon0
    )
    adapt_coords = list(zip(adapt_lats[::10], adapt_lons[::10]))
    folium.PolyLine(adapt_coords, color='#51cf66', weight=3, opacity=0.9,
                    tooltip='Adaptive System').add_to(m)
    
    # Current position marker (end of trajectory)
    folium.Marker(
        [adapt_lats[-1], adapt_lons[-1]],
        icon=folium.Icon(color='green', icon='car', prefix='fa'),
        tooltip='Current Position (Adaptive)',
    ).add_to(m)
    
    # Start marker
    folium.Marker(
        [gt_lats[0], gt_lons[0]],
        icon=folium.Icon(color='blue', icon='flag', prefix='fa'),
        tooltip='Start',
    ).add_to(m)
    
    # Legend
    legend_html = """
    <div style="position:fixed; bottom:30px; left:10px; z-index:1000;
         background:rgba(0,0,0,0.75); padding:10px; border-radius:8px;
         font-size:11px; color:white;">
    <b>Legend</b><br>
    <span style="color:#868e96;">━━</span> Ground Truth<br>
    <span style="color:#ffd43b;">╍╍</span> GNSS<br>
    <span style="color:#ff6b6b;">╍╍</span> Baseline INS<br>
    <span style="color:#51cf66;">━━</span> Adaptive System
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    st_folium(m, width=440, height=350, returned_objects=[])
    
    # ============================================================
    # NAVIGATION MODE
    # ============================================================
    render_section_title("📍 NAVIGATION STATUS")
    
    final_mode = adaptive_result['modes'][-1]
    render_mode_indicator(final_mode)
    
    # ============================================================
    # SENSOR CONFIDENCE
    # ============================================================
    render_section_title("📡 SENSOR CONFIDENCE")
    
    gnss_conf_final = adaptive_result['gnss_confidence'][-1] if len(adaptive_result['gnss_confidence']) > 0 else 1.0
    imu_conf_final = adaptive_result['imu_confidence'][-1] if len(adaptive_result['imu_confidence']) > 0 else 1.0
    overall_conf_final = adaptive_result['overall_confidence'][-1] if len(adaptive_result['overall_confidence']) > 0 else 1.0
    
    render_confidence_bar("GNSS", gnss_conf_final)
    render_confidence_bar("IMU", imu_conf_final)
    render_confidence_bar("OVERALL", overall_conf_final)
    
    # ============================================================
    # ACCURACY METRICS
    # ============================================================
    
    # Compute metrics for both methods
    gt_x = df['gt_x'].values
    gt_y = df['gt_y'].values
    
    # Baseline metrics
    n_baseline = min(len(baseline_traj['x']), len(gt_x))
    baseline_metrics = compute_full_metrics(
        baseline_traj['x'][:n_baseline], baseline_traj['y'][:n_baseline],
        gt_x[:n_baseline], gt_y[:n_baseline],
        estimated_speed=baseline_traj['speeds'][:n_baseline],
        truth_speed=df['gt_speed'].values[:n_baseline],
        blackout_mask=adaptive_result['blackout_mask'][:n_baseline],
    )
    baseline_metrics['method'] = 'Baseline INS'
    
    # Adaptive metrics
    n_adaptive = min(len(adaptive_result['x']), len(gt_x))
    adaptive_metrics = compute_full_metrics(
        adaptive_result['x'][:n_adaptive], adaptive_result['y'][:n_adaptive],
        gt_x[:n_adaptive], gt_y[:n_adaptive],
        estimated_speed=adaptive_result['speeds'][:n_adaptive],
        truth_speed=df['gt_speed'].values[:n_adaptive],
        blackout_mask=adaptive_result['blackout_mask'][:n_adaptive],
    )
    adaptive_metrics['method'] = 'Adaptive System'
    
    render_accuracy_panel(adaptive_metrics)
    
    # ============================================================
    # EVENTS
    # ============================================================
    render_events_panel(adaptive_result['events'])
    
    # Road disturbance summary
    road_summary = adaptive_result['road_detections']
    if road_summary['total_detections'] > 0:
        render_section_title("🛣️ ROAD DISTURBANCE")
        render_metric("Detections", str(road_summary['total_detections']))
        render_metric("Max Severity", f"{road_summary['max_severity']:.2f}")
        for cls, count in road_summary.get('classifications', {}).items():
            render_metric(cls, str(count))
    
    # ============================================================
    # COMPARISON TABLE
    # ============================================================
    render_section_title("📊 METHOD COMPARISON")
    
    st.plotly_chart(
        create_comparison_table([baseline_metrics, adaptive_metrics]),
        use_container_width=True,
    )
    
    # Blackout-specific comparison
    if 'blackout_rmse_m' in adaptive_metrics:
        render_section_title("🔴 GNSS BLACKOUT METRICS")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Baseline INS**")
            if 'blackout_rmse_m' in baseline_metrics:
                render_metric("RMSE", f"{baseline_metrics['blackout_rmse_m']:.2f} m")
                render_metric("Drift", f"{baseline_metrics.get('blackout_drift_percent', 0):.2f}%")
        with col2:
            st.markdown("**Adaptive**")
            render_metric("RMSE", f"{adaptive_metrics['blackout_rmse_m']:.2f} m")
            render_metric("Drift", f"{adaptive_metrics.get('blackout_drift_percent', 0):.2f}%")
    
    # ============================================================
    # TECHNICAL DETAILS
    # ============================================================
    with st.expander("🔬 Technical View — Plots & Analysis"):
        
        # Trajectory plot
        traj_fig = create_trajectory_plot(
            {
                'Ground Truth': {'x': gt_x, 'y': gt_y, 'color': '#868e96', 'dash': 'solid', 'width': 3},
                'Baseline INS': {'x': baseline_traj['x'][:n_baseline], 'y': baseline_traj['y'][:n_baseline],
                                 'color': '#ff6b6b', 'dash': 'dash', 'width': 2},
                'Adaptive': {'x': adaptive_result['x'][:n_adaptive], 'y': adaptive_result['y'][:n_adaptive],
                             'color': '#51cf66', 'dash': 'solid', 'width': 2},
            },
            blackout_start=blackout_start if blackout_start < 900 else None,
            blackout_end=blackout_end if blackout_end < 900 else None,
        )
        st.plotly_chart(traj_fig, use_container_width=True)
        
        # Error plot
        baseline_errors = position_error(
            baseline_traj['x'][:n_baseline], baseline_traj['y'][:n_baseline],
            gt_x[:n_baseline], gt_y[:n_baseline]
        )
        adaptive_errors = position_error(
            adaptive_result['x'][:n_adaptive], adaptive_result['y'][:n_adaptive],
            gt_x[:n_adaptive], gt_y[:n_adaptive]
        )
        
        error_fig = create_error_plot(
            timestamps[:n_baseline],
            {
                'Baseline INS': baseline_errors,
                'Adaptive': adaptive_errors[:n_baseline],
            },
            blackout_start=blackout_start if blackout_start < 900 else None,
            blackout_end=blackout_end if blackout_end < 900 else None,
        )
        st.plotly_chart(error_fig, use_container_width=True)
        
        # Confidence plot
        conf_timestamps = timestamps[1:len(adaptive_result['gnss_confidence'])+1]
        if len(conf_timestamps) > 0:
            conf_fig = create_confidence_plot(
                conf_timestamps,
                adaptive_result['gnss_confidence'][:len(conf_timestamps)],
                adaptive_result['imu_confidence'][:len(conf_timestamps)],
                adaptive_result['overall_confidence'][:len(conf_timestamps)],
                blackout_start=blackout_start if blackout_start < 900 else None,
                blackout_end=blackout_end if blackout_end < 900 else None,
            )
            st.plotly_chart(conf_fig, use_container_width=True)
        
        # Acceleration magnitude
        acc_fig = create_acceleration_plot(
            timestamps,
            df['acc_mag'].values if 'acc_mag' in df.columns else np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2).values,
            df['acc_mag_raw'].values if 'acc_mag_raw' in df.columns else None,
            blackout_start=blackout_start if blackout_start < 900 else None,
            blackout_end=blackout_end if blackout_end < 900 else None,
        )
        st.plotly_chart(acc_fig, use_container_width=True)
        
        # Speed plot
        speed_fig = create_speed_plot(
            timestamps,
            {
                'Ground Truth': df['gt_speed'].values,
                'Baseline INS': baseline_traj['speeds'][:n_baseline],
                'Adaptive': adaptive_result['speeds'][:n_adaptive],
            },
            blackout_start=blackout_start if blackout_start < 900 else None,
            blackout_end=blackout_end if blackout_end < 900 else None,
        )
        st.plotly_chart(speed_fig, use_container_width=True)
        
        # IMU raw vs processed
        render_section_title("IMU Data: Raw vs Processed")
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        imu_fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                 subplot_titles=('Acc X (forward)', 'Acc Y (lateral)', 'Acc Z (vertical)'))
        
        for idx, axis in enumerate(['acc_x', 'acc_y', 'acc_z']):
            raw_col = f'{axis}_raw'
            if raw_col in df.columns:
                imu_fig.add_trace(go.Scatter(
                    x=timestamps, y=df[raw_col].values,
                    mode='lines', name=f'{axis} raw',
                    line=dict(color='rgba(255,107,107,0.3)', width=1),
                    showlegend=(idx == 0),
                ), row=idx+1, col=1)
            
            imu_fig.add_trace(go.Scatter(
                x=timestamps, y=df[axis].values,
                mode='lines', name=f'{axis} processed',
                line=dict(color='#51cf66', width=1),
                showlegend=(idx == 0),
            ), row=idx+1, col=1)
        
        imu_fig.update_layout(
            height=500, template="plotly_dark",
            title="IMU Data: Raw vs Processed",
        )
        st.plotly_chart(imu_fig, use_container_width=True)
    
    # ============================================================
    # DATASET INFO
    # ============================================================
    with st.expander("📁 Dataset & Preprocessing Info"):
        st.markdown("⚠️ **SIMULATED DATA — NOT REAL EXPERIMENTAL RESULTS**")
        st.markdown(f"""
        - **Duration**: {duration:.1f}s
        - **IMU Rate**: {meta.get('imu_rate', 100)} Hz
        - **GNSS Rate**: {meta.get('gnss_rate', 10)} Hz
        - **Samples**: {len(df)}
        - **Origin**: {lat0:.6f}°N, {lon0:.6f}°E (Hyderabad)
        """)
        
        if prep_info:
            st.markdown("**Preprocessing:**")
            st.markdown(f"- Duplicates removed: {prep_info.get('duplicates_removed', 0)}")
            st.markdown(f"- Missing values filled: {prep_info.get('missing_values_filled', 0)}")
            if 'acc_bias' in prep_info:
                bias = prep_info['acc_bias']
                st.markdown(f"- Acc bias: x={bias['acc_x']:.4f}, y={bias['acc_y']:.4f}, z={bias['acc_z']:.4f} m/s²")
            if 'gyro_bias' in prep_info:
                bias = prep_info['gyro_bias']
                st.markdown(f"- Gyro bias: x={bias['gyro_x']:.6f}, y={bias['gyro_y']:.6f}, z={bias['gyro_z']:.6f} rad/s")
    
    # ============================================================
    # FOOTER
    # ============================================================
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#4b5563; font-size:11px; padding:8px;">
        SIH Problem Statement 168 — AI-ML Intelligent Dead Reckoning<br>
        Prototype v1.0 | Simulated Data | Phase 2: EKF, AI Velocity, Offline Maps<br>
        <em>"Continuously estimates which sensors can be trusted and adapts accordingly."</em>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
