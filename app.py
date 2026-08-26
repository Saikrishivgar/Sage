"""
SAGE — Smart Adaptive Guidance Engine
=======================================
Adaptive GNSS–IMU Navigation System

Continuously estimates position using smartphone IMU sensors when
GNSS becomes unreliable or unavailable.

SIMULATED DATA — Prototype v1.0
Rule-based confidence engine — ML replacement planned for Phase 2.

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
    apply_sage_style, render_sage_header, render_section_title,
    render_mode_indicator, render_confidence_bar, render_metric,
    render_key_metric, render_events_panel, render_live_position,
    render_scenario_card, render_pipeline_diagram, render_sim_badge,
    render_accuracy_panel, render_footer,
)

st.set_page_config(
    page_title="SAGE — Adaptive Navigation",
    page_icon="🧭",
    layout="centered",
    initial_sidebar_state="collapsed",
)

apply_sage_style()


# ============================================================
# PIPELINE FUNCTIONS (unchanged backend logic)
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

    fusion = AdaptiveFusion()
    gnss_detector = GNSSAnomalyDetector()
    road_detector = RoadDisturbanceDetector(fs=meta.get('imu_rate', 100))

    dt_arr = compute_dt(df)

    fused_x = [df['gt_x'].iloc[0]]
    fused_y = [df['gt_y'].iloc[0]]
    fused_speeds = [0.0]
    navigation_modes = ["GNSS + INS"]
    blackout_mask = [False]

    current_pos = np.array([df['gt_x'].iloc[0], df['gt_y'].iloc[0]])
    current_heading = float(df['gt_heading'].iloc[0])
    current_speed = 0.0
    last_gnss_pos = current_pos.copy()
    last_gnss_heading = current_heading
    last_gnss_speed = 0.0

    gnss_outage_start_time = None
    dr_velocity = np.array([0.0, 0.0])

    for i in range(1, len(df)):
        t = df['timestamp'].iloc[i]
        dt = dt_arr[i]

        acc_fwd = df['acc_x'].iloc[i]
        acc_lat = df['acc_y'].iloc[i]
        gyro_z = df['gyro_z'].iloc[i]

        current_heading += gyro_z * dt
        current_heading = current_heading % (2 * np.pi)

        gnss_available = not (blackout_start <= t <= blackout_end)
        is_blackout = (blackout_start <= t <= blackout_end)
        blackout_mask.append(is_blackout)

        if not gnss_available:
            if gnss_outage_start_time is None:
                gnss_outage_start_time = t
                dr_velocity = np.array([
                    last_gnss_speed * np.sin(current_heading),
                    last_gnss_speed * np.cos(current_heading),
                ])
            integration_time = t - gnss_outage_start_time
        else:
            gnss_outage_start_time = None
            integration_time = 0.0

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

            if gnss_anomaly_time is not None and abs(t - gnss_anomaly_time) < 2.0:
                gnss_lat += 0.0005
                gnss_lon += 0.0005

            gx, gy = latlon_to_local(
                np.array([gnss_lat]), np.array([gnss_lon]), lat0, lon0
            )
            gnss_position = np.array([gx[0], gy[0]])
            gnss_speed_val = float(gnss_speed_val)
            gnss_heading_val = float(df['gnss_heading'].iloc[i]) if 'gnss_heading' in df.columns else None

            anomaly_result = gnss_detector.detect(
                gnss_lat, gnss_lon, gnss_speed_val, hdop,
                current_pos[0], current_pos[1], t, lat0, lon0
            )
            gnss_anomaly = anomaly_result['anomaly_detected']

            if not gnss_anomaly:
                last_gnss_pos = gnss_position.copy()
                last_gnss_heading = gnss_heading_val if gnss_heading_val is not None else current_heading
                last_gnss_speed = gnss_speed_val

        if not gnss_available:
            sin_h = np.sin(current_heading)
            cos_h = np.cos(current_heading)
            acc_east = acc_fwd * sin_h + acc_lat * cos_h
            acc_north = acc_fwd * cos_h - acc_lat * sin_h
            dr_velocity[0] += acc_east * dt
            dr_velocity[1] += acc_north * dt

            dr_speed = np.linalg.norm(dr_velocity)
            if dr_speed > 50.0:
                dr_velocity *= 50.0 / dr_speed

            ins_position = current_pos + dr_velocity * dt
            ins_speed = np.linalg.norm(dr_velocity)
        else:
            ins_speed = last_gnss_speed
            ins_position = current_pos + np.array([
                ins_speed * np.sin(current_heading),
                ins_speed * np.cos(current_heading),
            ]) * dt

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

    fused_x = np.array(fused_x)
    fused_y = np.array(fused_y)
    fused_speeds = np.array(fused_speeds)
    blackout_mask = np.array(blackout_mask)

    conf_history = fusion.confidence.history
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
# HELPER: Create Folium Map
# ============================================================

def create_map(df, meta, adaptive_result, baseline_traj, height=400):
    """Create a Folium map with all trajectories."""
    lat0 = meta['lat0']
    lon0 = meta['lon0']

    m = folium.Map(
        location=[lat0, lon0],
        zoom_start=16,
        tiles='CartoDB positron',
        attr='© CartoDB'
    )

    # Ground truth
    gt_lats = df['gt_lat'].values
    gt_lons = df['gt_lon'].values
    gt_coords = list(zip(gt_lats[::10], gt_lons[::10]))
    folium.PolyLine(gt_coords, color='#5f6368', weight=3, opacity=0.5,
                    tooltip='Ground Truth', dash_array='6').add_to(m)

    # GNSS raw
    gnss_mask = ~df['gnss_lat'].isna()
    if gnss_mask.any():
        gnss_lats = df.loc[gnss_mask, 'gnss_lat'].values
        gnss_lons = df.loc[gnss_mask, 'gnss_lon'].values
        gnss_coords = list(zip(gnss_lats[::5], gnss_lons[::5]))
        if len(gnss_coords) > 1:
            folium.PolyLine(gnss_coords, color='#fbbc04', weight=2, opacity=0.4,
                            dash_array='4', tooltip='GNSS Raw').add_to(m)

    # Baseline INS
    baseline_lats, baseline_lons = local_to_latlon(
        baseline_traj['x'], baseline_traj['y'], lat0, lon0
    )
    baseline_coords = list(zip(baseline_lats[::10], baseline_lons[::10]))
    folium.PolyLine(baseline_coords, color='#ea4335', weight=2, opacity=0.5,
                    dash_array='8', tooltip='Baseline INS (drift)').add_to(m)

    # SAGE Adaptive
    adapt_lats, adapt_lons = local_to_latlon(
        adaptive_result['x'], adaptive_result['y'], lat0, lon0
    )
    adapt_coords = list(zip(adapt_lats[::10], adapt_lons[::10]))
    folium.PolyLine(adapt_coords, color='#34a853', weight=3, opacity=0.9,
                    tooltip='SAGE Adaptive').add_to(m)

    # Current position marker
    folium.Marker(
        [adapt_lats[-1], adapt_lons[-1]],
        icon=folium.Icon(color='green', icon='location-dot', prefix='fa'),
        tooltip='SAGE Position',
    ).add_to(m)

    # Start marker
    folium.Marker(
        [gt_lats[0], gt_lons[0]],
        icon=folium.Icon(color='blue', icon='flag', prefix='fa'),
        tooltip='Start',
    ).add_to(m)

    # Light legend
    legend_html = """
    <div style="position:fixed; bottom:30px; left:10px; z-index:1000;
         background:white; padding:10px 14px; border-radius:8px;
         font-size:11px; color:#3c4043; box-shadow:0 1px 4px rgba(0,0,0,0.15);
         border:1px solid #e5e7eb; font-family:Inter,sans-serif;">
    <b style="font-size:12px;">Legend</b><br>
    <span style="color:#5f6368;">╍╍</span> Ground Truth<br>
    <span style="color:#fbbc04;">╍╍</span> GNSS Raw<br>
    <span style="color:#ea4335;">╍╍</span> Baseline INS<br>
    <span style="color:#34a853;">━━</span> <b>SAGE Adaptive</b>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m, adapt_lats, adapt_lons


# ============================================================
# MAIN APP
# ============================================================

def main():
    render_sage_header()

    # Load data
    try:
        df, meta, prep_info = load_and_preprocess()
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        st.info("Run `python generate_dataset.py` first to create the simulated dataset.")
        return

    timestamps = df['timestamp'].values
    duration = timestamps[-1] - timestamps[0]
    lat0 = meta['lat0']
    lon0 = meta['lon0']

    # ============================================================
    # SCENARIO STATE (persists across tabs)
    # ============================================================
    if 'scenario' not in st.session_state:
        st.session_state.scenario = 'normal'

    scenario = st.session_state.scenario

    # Scenario parameters
    scenarios = {
        'normal': {
            'blackout_start': 999, 'blackout_end': 999,
            'disturbance_enabled': False, 'gnss_anomaly_time': None,
            'label': 'Normal — Full GNSS + INS',
        },
        'gnss_loss': {
            'blackout_start': 30, 'blackout_end': 60,
            'disturbance_enabled': False, 'gnss_anomaly_time': None,
            'label': 'GNSS Loss — 30s blackout (30s–60s)',
        },
        'gnss_jump': {
            'blackout_start': 999, 'blackout_end': 999,
            'disturbance_enabled': False, 'gnss_anomaly_time': 45.0,
            'label': 'GNSS Jump — ~55m anomaly at 45s',
        },
        'rough_road': {
            'blackout_start': 999, 'blackout_end': 999,
            'disturbance_enabled': True, 'gnss_anomaly_time': None,
            'label': 'Rough Road — Disturbance detection',
        },
        'combined': {
            'blackout_start': 30, 'blackout_end': 60,
            'disturbance_enabled': True, 'gnss_anomaly_time': None,
            'label': 'Combined — GNSS loss + Rough road',
        },
    }

    if scenario == 'custom' and 'custom_params' in st.session_state:
        params = st.session_state.custom_params
    else:
        params = scenarios.get(scenario, scenarios['normal'])
    blackout_start = params['blackout_start']
    blackout_end = params['blackout_end']
    disturbance_enabled = params['disturbance_enabled']
    gnss_anomaly_time = params['gnss_anomaly_time']
    scenario_label = params['label']

    # ============================================================
    # RUN PIPELINES
    # ============================================================
    baseline_traj = run_baseline_dr(df, meta)

    adaptive_result = run_adaptive_pipeline(
        df, meta, blackout_start, blackout_end,
        disturbance_enabled=disturbance_enabled,
        gnss_anomaly_time=gnss_anomaly_time,
    )

    # Compute metrics
    gt_x = df['gt_x'].values
    gt_y = df['gt_y'].values

    n_baseline = min(len(baseline_traj['x']), len(gt_x))
    baseline_metrics = compute_full_metrics(
        baseline_traj['x'][:n_baseline], baseline_traj['y'][:n_baseline],
        gt_x[:n_baseline], gt_y[:n_baseline],
        estimated_speed=baseline_traj['speeds'][:n_baseline],
        truth_speed=df['gt_speed'].values[:n_baseline],
        blackout_mask=adaptive_result['blackout_mask'][:n_baseline],
    )
    baseline_metrics['method'] = 'Baseline INS'

    n_adaptive = min(len(adaptive_result['x']), len(gt_x))
    adaptive_metrics = compute_full_metrics(
        adaptive_result['x'][:n_adaptive], adaptive_result['y'][:n_adaptive],
        gt_x[:n_adaptive], gt_y[:n_adaptive],
        estimated_speed=adaptive_result['speeds'][:n_adaptive],
        truth_speed=df['gt_speed'].values[:n_adaptive],
        blackout_mask=adaptive_result['blackout_mask'][:n_adaptive],
    )
    adaptive_metrics['method'] = 'SAGE Adaptive'

    # Final state values
    final_mode = adaptive_result['modes'][-1]
    gnss_conf_final = adaptive_result['gnss_confidence'][-1] if len(adaptive_result['gnss_confidence']) > 0 else 1.0
    imu_conf_final = adaptive_result['imu_confidence'][-1] if len(adaptive_result['imu_confidence']) > 0 else 1.0
    overall_conf_final = adaptive_result['overall_confidence'][-1] if len(adaptive_result['overall_confidence']) > 0 else 1.0

    # Lat/lon of final SAGE position
    adapt_lats, adapt_lons = local_to_latlon(
        adaptive_result['x'], adaptive_result['y'], lat0, lon0
    )
    final_lat = adapt_lats[-1]
    final_lon = adapt_lons[-1]
    final_speed = adaptive_result['speeds'][-1]
    final_heading = df['gt_heading'].iloc[-1]
    final_error = adaptive_metrics.get('final_error_m', 0)

    # ============================================================
    # 5-TAB LAYOUT
    # ============================================================
    tab_home, tab_nav, tab_test, tab_insights, tab_tech = st.tabs([
        "🏠 Home",
        "🧭 Navigate",
        "🧪 Test Lab",
        "📊 Insights",
        "🔬 Technical",
    ])

    # ── TAB 1: HOME ─────────────────────────────────────────
    with tab_home:
        render_sim_badge()

        # Map
        m, _, _ = create_map(df, meta, adaptive_result, baseline_traj, height=380)
        st_folium(m, width=700, height=380, returned_objects=[], key="map_home")

        # Navigation Status
        render_section_title("Navigation Status")
        render_mode_indicator(final_mode)

        # Sensor Trust
        render_section_title("Sensor Trust")
        col1, col2 = st.columns(2)
        with col1:
            render_confidence_bar("GNSS", gnss_conf_final)
        with col2:
            render_confidence_bar("IMU", imu_conf_final)

        # Quick Stats
        render_section_title("Quick Status")
        col1, col2, col3 = st.columns(3)
        with col1:
            render_key_metric(
                "Accuracy",
                f"{final_error:.1f} m",
                "pass" if final_error < 10 else "fail"
            )
        with col2:
            render_key_metric("Drift", f"{adaptive_metrics.get('drift_percent', 0):.2f}%")
        with col3:
            render_key_metric(
                "Status",
                adaptive_metrics.get('pass_fail', 'N/A'),
                "pass" if adaptive_metrics.get('pass_fail') == 'PASS' else "fail"
            )

        st.markdown(f"""
        <div style="text-align:center; margin-top:12px; color:#80868b; font-size:12px;">
            Active Scenario: <strong>{scenario_label}</strong>
        </div>
        """, unsafe_allow_html=True)

    # ── TAB 2: LIVE NAVIGATION ───────────────────────────────
    with tab_nav:
        render_sim_badge()

        # Large Map
        m_nav, _, _ = create_map(df, meta, adaptive_result, baseline_traj, height=450)
        st_folium(m_nav, width=700, height=450, returned_objects=[], key="map_nav")

        # Live Position
        render_section_title("Live Position")
        render_live_position(final_lat, final_lon, final_speed,
                             final_heading, final_error)

        # Navigation Mode (dramatic)
        render_section_title("Navigation Mode")
        render_mode_indicator(final_mode)

        # GNSS alert
        if "DEAD RECKONING" in final_mode:
            st.markdown(
                '<div class="alert-banner alert-gnss-lost">'
                '🔴 GNSS SIGNAL LOST — Estimating position via IMU dead reckoning'
                '</div>',
                unsafe_allow_html=True
            )
        elif gnss_conf_final < 0.5:
            st.markdown(
                '<div class="alert-banner alert-gnss-lost">'
                '⚠️ GNSS DEGRADED — Confidence below 50%'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="alert-banner alert-gnss-ok">'
                '🟢 All navigation sensors active and healthy'
                '</div>',
                unsafe_allow_html=True
            )

        # Sensor Confidence
        render_section_title("Sensor Confidence")
        render_confidence_bar("GNSS", gnss_conf_final)
        render_confidence_bar("IMU", imu_conf_final)
        render_confidence_bar("Overall", overall_conf_final)

        # Events
        render_events_panel(adaptive_result['events'])

    # ── TAB 3: TEST LAB ──────────────────────────────────────
    with tab_test:
        render_section_title("🧪 Select a Navigation Scenario")

        st.markdown("""
        <div style="color:#5f6368; font-size:13px; margin-bottom:12px;">
        Choose a failure scenario to stress-test SAGE's adaptive navigation.
        The system will automatically adjust sensor confidence and switch
        navigation modes in real-time.
        </div>
        """, unsafe_allow_html=True)

        # Scenario selection
        scenario_defs = [
            ('normal', '🟢 Normal', 'Baseline navigation with full GNSS + INS',
             'Stable navigation, low error, high confidence'),
            ('gnss_loss', '🔴 GNSS Loss', '30-second GPS blackout (30s – 60s)',
             'GNSS conf → 0%, mode → Dead Reckoning, IMU takes over'),
            ('gnss_jump', '🟠 GNSS Jump', '~55m position anomaly injected at 45s',
             'Anomaly detector flags jump, GNSS conf drops, fusion rejects bad fix'),
            ('rough_road', '🟡 Rough Road', 'Road disturbance spikes (42s – 48s)',
             'IMU conf reduces during spikes, fusion adapts weights'),
            ('combined', '💥 Combined', 'GPS loss (30s–60s) + rough road disturbance',
             'Multiple failures: DR mode + disturbance → max stress test'),
        ]

        cols = st.columns(3)
        for idx, (key, title, desc, expected) in enumerate(scenario_defs[:3]):
            with cols[idx]:
                is_active = scenario == key
                render_scenario_card(title, "", desc, expected, is_active)
                if st.button(f"Run {title}", key=f"btn_{key}", use_container_width=True):
                    st.session_state.scenario = key
                    st.rerun()

        cols2 = st.columns(3)
        for idx, (key, title, desc, expected) in enumerate(scenario_defs[3:]):
            with cols2[idx]:
                is_active = scenario == key
                render_scenario_card(title, "", desc, expected, is_active)
                if st.button(f"Run {title}", key=f"btn_{key}", use_container_width=True):
                    st.session_state.scenario = key
                    st.rerun()

        # Active scenario result
        render_section_title("Scenario Result")
        st.markdown(f"""
        <div style="text-align:center; color:#3c4043; font-size:14px; margin:8px 0;">
            <strong>Active:</strong> {scenario_label}
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            render_key_metric("Final Error", f"{final_error:.1f} m",
                              "pass" if final_error < 10 else "fail")
        with col2:
            render_key_metric("Drift", f"{adaptive_metrics.get('drift_percent', 0):.2f}%")
        with col3:
            render_key_metric("Final Mode", final_mode.split('(')[0].strip())

        # Road disturbance summary
        road_summary = adaptive_result['road_detections']
        if road_summary['total_detections'] > 0:
            render_section_title("Road Disturbance Detected")
            render_metric("Detections", str(road_summary['total_detections']))
            render_metric("Max Severity", f"{road_summary['max_severity']:.2f}")
            for cls, count in road_summary.get('classifications', {}).items():
                render_metric(cls, str(count))

        # Custom scenario
        with st.expander("⚙️ Custom Scenario"):
            custom_start = st.slider("GNSS Outage Start (s)", 0, int(duration) - 5,
                                     int(blackout_start) if blackout_start < 900 else 30)
            custom_duration = st.slider("GNSS Outage Duration (s)", 5, 60,
                                        int(blackout_end - blackout_start) if blackout_start < 900 else 30)
            custom_disturbance = st.checkbox("Enable Road Disturbance Detection",
                                            value=disturbance_enabled)
            if st.button("Apply Custom"):
                st.session_state.scenario = 'custom'
                st.session_state.custom_params = {
                    'blackout_start': custom_start,
                    'blackout_end': custom_start + custom_duration,
                    'disturbance_enabled': custom_disturbance,
                    'gnss_anomaly_time': None,
                    'label': f'Custom — Blackout {custom_start}s–{custom_start + custom_duration}s',
                }
                st.rerun()

        # Dataset info
        render_section_title("📁 Dataset Info")
        render_sim_badge()
        render_metric("Duration", f"{duration:.1f}s")
        render_metric("IMU Rate", f"{meta.get('imu_rate', 100)} Hz")
        render_metric("GNSS Rate", f"{meta.get('gnss_rate', 10)} Hz")
        render_metric("Samples", str(len(df)))
        render_metric("Origin", f"{lat0:.4f}°N, {lon0:.4f}°E")

    # ── TAB 4: NAVIGATION INSIGHTS ───────────────────────────
    with tab_insights:
        render_section_title("Key Metrics")

        col1, col2 = st.columns(2)
        with col1:
            render_key_metric(
                "Position Accuracy",
                f"{final_error:.1f} m",
                "pass" if final_error < 10 else "fail"
            )
        with col2:
            render_key_metric("RMSE", f"{adaptive_metrics.get('rmse_m', 0):.1f} m")

        col3, col4 = st.columns(2)
        with col3:
            render_key_metric("Max Error", f"{adaptive_metrics.get('max_error_m', 0):.1f} m")
        with col4:
            drift = adaptive_metrics.get('drift_percent', 0)
            render_key_metric("Drift", f"{drift:.2f}%",
                              "pass" if drift < 10 else "fail")

        # Pass/Fail badge
        status = adaptive_metrics.get('pass_fail', 'N/A')
        st.markdown(f"""
        <div style="text-align:center; margin:12px 0;">
            <span style="display:inline-block; padding:6px 20px; border-radius:20px;
                         font-weight:700; font-size:14px;
                         background:{'#e6f4ea' if status == 'PASS' else '#fce8e6'};
                         color:{'#137333' if status == 'PASS' else '#c5221f'};
                         border:1px solid {'#ceead6' if status == 'PASS' else '#f5c6c4'};">
                {status}
            </span>
        </div>
        """, unsafe_allow_html=True)

        # SAGE vs Baseline table
        render_section_title("SAGE vs Baseline")
        st.plotly_chart(
            create_comparison_table([baseline_metrics, adaptive_metrics]),
            use_container_width=True,
        )

        # Blackout-specific comparison
        if 'blackout_rmse_m' in adaptive_metrics:
            render_section_title("GNSS Blackout Performance")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Baseline INS**")
                if 'blackout_rmse_m' in baseline_metrics:
                    render_metric("RMSE", f"{baseline_metrics['blackout_rmse_m']:.2f} m")
                    render_metric("Drift", f"{baseline_metrics.get('blackout_drift_percent', 0):.2f}%")
            with col2:
                st.markdown("**SAGE Adaptive**")
                render_metric("RMSE", f"{adaptive_metrics['blackout_rmse_m']:.2f} m")
                render_metric("Drift", f"{adaptive_metrics.get('blackout_drift_percent', 0):.2f}%")

        # Charts
        render_section_title("Position Error Over Time")

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
                'SAGE': adaptive_errors[:n_baseline],
            },
            blackout_start=blackout_start if blackout_start < 900 else None,
            blackout_end=blackout_end if blackout_end < 900 else None,
        )
        st.plotly_chart(error_fig, use_container_width=True)

        # Confidence over time
        render_section_title("Sensor Confidence Over Time")
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

        # Speed comparison
        render_section_title("Speed Comparison")
        speed_fig = create_speed_plot(
            timestamps,
            {
                'Ground Truth': df['gt_speed'].values,
                'Baseline INS': baseline_traj['speeds'][:n_baseline],
                'SAGE': adaptive_result['speeds'][:n_adaptive],
            },
            blackout_start=blackout_start if blackout_start < 900 else None,
            blackout_end=blackout_end if blackout_end < 900 else None,
        )
        st.plotly_chart(speed_fig, use_container_width=True)

        # Advanced diagnostics
        with st.expander("Advanced Diagnostics"):
            render_metric("Velocity MAE",
                          f"{adaptive_metrics.get('velocity_mae_ms', 0):.2f} m/s")
            render_metric("Velocity RMSE",
                          f"{adaptive_metrics.get('velocity_rmse_ms', 0):.2f} m/s")
            render_metric("Est. Distance",
                          f"{adaptive_metrics.get('estimated_distance_m', 0):.0f} m")
            render_metric("True Distance",
                          f"{adaptive_metrics.get('distance_travelled_m', 0):.0f} m")

    # ── TAB 5: TECHNICAL ─────────────────────────────────────
    with tab_tech:
        render_section_title("System Pipeline")
        render_pipeline_diagram()

        st.markdown("""
        <div style="color:#5f6368; font-size:13px; margin:8px 0; line-height:1.6;">
            SAGE processes raw IMU data through bias correction and low-pass filtering,
            then feeds it into the adaptive fusion engine alongside GNSS measurements.
            Sensor confidence values are computed in real-time using rule-based heuristics
            (ML-based prediction planned for Phase 2). The fusion engine weights GNSS and
            INS estimates proportionally to their confidence.
        </div>
        """, unsafe_allow_html=True)

        # Trajectory comparison
        render_section_title("Trajectory Comparison")
        traj_fig = create_trajectory_plot(
            {
                'Ground Truth': {'x': gt_x, 'y': gt_y, 'color': '#5f6368', 'dash': 'dash', 'width': 2},
                'Baseline INS': {'x': baseline_traj['x'][:n_baseline], 'y': baseline_traj['y'][:n_baseline],
                                 'color': '#ea4335', 'dash': 'dash', 'width': 2},
                'SAGE': {'x': adaptive_result['x'][:n_adaptive], 'y': adaptive_result['y'][:n_adaptive],
                         'color': '#34a853', 'dash': 'solid', 'width': 2},
            },
            blackout_start=blackout_start if blackout_start < 900 else None,
            blackout_end=blackout_end if blackout_end < 900 else None,
        )
        st.plotly_chart(traj_fig, use_container_width=True)

        # Acceleration magnitude
        render_section_title("Acceleration Magnitude")
        acc_fig = create_acceleration_plot(
            timestamps,
            df['acc_mag'].values if 'acc_mag' in df.columns else np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2).values,
            df['acc_mag_raw'].values if 'acc_mag_raw' in df.columns else None,
            blackout_start=blackout_start if blackout_start < 900 else None,
            blackout_end=blackout_end if blackout_end < 900 else None,
        )
        st.plotly_chart(acc_fig, use_container_width=True)

        # IMU raw vs processed
        render_section_title("IMU Data: Raw vs Processed")
        import plotly.graph_objects as go_tech
        from plotly.subplots import make_subplots

        imu_fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                 subplot_titles=('Acc X (forward)', 'Acc Y (lateral)', 'Acc Z (vertical)'))

        for idx, axis in enumerate(['acc_x', 'acc_y', 'acc_z']):
            raw_col = f'{axis}_raw'
            if raw_col in df.columns:
                imu_fig.add_trace(go_tech.Scatter(
                    x=timestamps, y=df[raw_col].values,
                    mode='lines', name=f'{axis} raw',
                    line=dict(color='rgba(234,67,53,0.25)', width=1),
                    showlegend=(idx == 0),
                ), row=idx+1, col=1)

            imu_fig.add_trace(go_tech.Scatter(
                x=timestamps, y=df[axis].values,
                mode='lines', name=f'{axis} processed',
                line=dict(color='#34a853', width=1),
                showlegend=(idx == 0),
            ), row=idx+1, col=1)

        imu_fig.update_layout(
            height=500, template="plotly_white",
            title="IMU Data: Raw vs Processed",
            font=dict(family="Inter, sans-serif", size=12, color="#3c4043"),
        )
        st.plotly_chart(imu_fig, use_container_width=True)

        # Preprocessing info
        render_section_title("Preprocessing Details")
        if prep_info:
            render_metric("Duplicates Removed", str(prep_info.get('duplicates_removed', 0)))
            render_metric("Missing Values Filled", str(prep_info.get('missing_values_filled', 0)))
            if 'acc_bias' in prep_info:
                bias = prep_info['acc_bias']
                render_metric("Acc Bias (x, y, z)",
                              f"{bias['acc_x']:.4f}, {bias['acc_y']:.4f}, {bias['acc_z']:.4f} m/s²")
            if 'gyro_bias' in prep_info:
                bias = prep_info['gyro_bias']
                render_metric("Gyro Bias (x, y, z)",
                              f"{bias['gyro_x']:.6f}, {bias['gyro_y']:.6f}, {bias['gyro_z']:.6f} rad/s")
            if 'filter' in prep_info:
                f_info = prep_info['filter']
                render_metric("Filter", f"{f_info['type']} order {f_info['order']}, cutoff {f_info['cutoff_hz']}Hz")

    # ── FOOTER ───────────────────────────────────────────────
    render_footer()


if __name__ == '__main__':
    main()
