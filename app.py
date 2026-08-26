"""
SAGE — Smart Adaptive Guidance Engine
=======================================
Adaptive GNSS–IMU Navigation System

Continuously estimates position using smartphone IMU sensors when
GNSS becomes unreliable or unavailable.

SIMULATED DATA — FOR DEMONSTRATION
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
from src.simulation.simulator import GNSSFailureSimulator, ENVIRONMENTS
import time
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

            # Support single float or list of anomaly times
            anomaly_times = (
                gnss_anomaly_time if isinstance(gnss_anomaly_time, list)
                else ([gnss_anomaly_time] if gnss_anomaly_time is not None else [])
            )
            if any(abs(t - at) < 2.0 for at in anomaly_times):
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
# HELPER: Compute Representative Scenario State
# ============================================================

def compute_representative_state(adaptive_result, scenario, blackout_start,
                                 blackout_end, gnss_anomaly_time, timestamps):
    """
    Extract the representative state for display instead of the final timestep.

    For Normal: use final state (everything healthy).
    For all failure scenarios: use the moment of LOWEST overall confidence,
    which naturally corresponds to the most dramatic event point.

    This ensures GNSS Loss shows GNSS=0%, not post-recovery GNSS=97%.
    """
    gnss_conf = adaptive_result['gnss_confidence']
    imu_conf = adaptive_result['imu_confidence']
    overall_conf = adaptive_result['overall_confidence']
    modes = adaptive_result['modes']

    if scenario == 'normal':
        # Normal: everything is healthy — use final state
        idx = len(gnss_conf) - 1
    else:
        # Failure scenarios: find the moment of minimum overall confidence
        # This naturally captures the dramatic event (blackout mid-point,
        # anomaly detection, disturbance peak, or combined worst-case)
        idx = int(np.argmin(overall_conf))

    # +1 offset because confidence arrays start from step 1, modes from step 0
    mode_idx = min(idx + 1, len(modes) - 1)

    return {
        'gnss_conf': float(gnss_conf[idx]),
        'imu_conf': float(imu_conf[idx]),
        'overall_conf': float(overall_conf[idx]),
        'mode': modes[mode_idx],
        'index': idx,
    }


def build_scenario_events(scenario, rep_state, road_summary):
    """
    Build scenario-aware event messages for display.
    These are deterministic and tied to the actual scenario state.
    """
    events = []
    gnss_c = rep_state['gnss_conf']
    imu_c = rep_state['imu_conf']
    mode = rep_state['mode']

    if scenario == 'normal':
        events.append("✅ All navigation sensors active and healthy")
        events.append(f"✅ GNSS confidence: {gnss_c*100:.0f}%")
        events.append(f"✅ IMU confidence: {imu_c*100:.0f}%")
        events.append(f"✅ Navigation mode: {mode}")

    elif scenario == 'gnss_loss':
        events.append("🔴 GNSS SIGNAL LOST — Satellite signal unavailable")
        events.append(f"🔴 GNSS confidence: {gnss_c*100:.0f}%")
        events.append(f"✅ IMU confidence: {imu_c*100:.0f}% — IMU operational")
        events.append(f"🔴 Navigation mode: {mode}")
        events.append("⚠️ Dead reckoning active — position estimated via IMU integration")
        events.append("⚠️ Position accuracy degrades over time without GNSS corrections")

    elif scenario == 'gnss_jump':
        events.append("⚠️ GNSS anomaly detected — sudden ~55m position jump")
        events.append(f"⚠️ GNSS confidence reduced: {gnss_c*100:.0f}%")
        events.append(f"✅ IMU confidence: {imu_c*100:.0f}% — IMU weighted higher")
        events.append("✅ SAGE rejected anomalous GNSS measurement")
        events.append(f"✅ Navigation mode: {mode}")

    elif scenario == 'rough_road':
        events.append("⚠️ Road disturbance detected — abnormal acceleration spikes")
        events.append(f"✅ GNSS confidence: {gnss_c*100:.0f}%")
        events.append(f"⚠️ IMU confidence reduced: {imu_c*100:.0f}%")
        events.append("⚠️ Sensor weights adjusted — GNSS weighted higher during disturbance")
        if road_summary['total_detections'] > 0:
            events.append(f"⚠️ {road_summary['total_detections']} disturbance events detected")
            for cls, count in road_summary.get('classifications', {}).items():
                events.append(f"   • {cls}: {count}")

    elif scenario == 'combined':
        events.append("🔴 GNSS SIGNAL LOST + ROAD DISTURBANCE")
        events.append(f"🔴 GNSS confidence: {gnss_c*100:.0f}%")
        events.append(f"⚠️ IMU confidence reduced: {imu_c*100:.0f}%")
        events.append(f"🔴 Navigation mode: {mode}")
        events.append("⚠️ Both primary sensors degraded — SAGE using best available estimate")
        if road_summary['total_detections'] > 0:
            events.append(f"⚠️ {road_summary['total_detections']} disturbance events during GNSS outage")

    else:  # custom
        if gnss_c < 0.1:
            events.append("🔴 GNSS SIGNAL LOST")
        elif gnss_c < 0.5:
            events.append(f"⚠️ GNSS degraded: {gnss_c*100:.0f}%")
        else:
            events.append(f"✅ GNSS: {gnss_c*100:.0f}%")
        events.append(f"{'⚠️' if imu_c < 0.8 else '✅'} IMU: {imu_c*100:.0f}%")
        events.append(f"Navigation mode: {mode}")

    return events


# ============================================================
# HELPER: Build Scenario Status Descriptor
# ============================================================

def get_scenario_status_text(scenario, rep_state):
    """Return a concise status descriptor for the active scenario."""
    if scenario == 'normal':
        return "🟢 All sensors healthy — Full GNSS + INS navigation"
    elif scenario == 'gnss_loss':
        return "🔴 GNSS unavailable — Dead reckoning active"
    elif scenario == 'gnss_jump':
        return "🟡 GNSS anomaly detected — IMU weighted higher"
    elif scenario == 'rough_road':
        return "🟡 Road disturbance detected — sensor weights adjusted"
    elif scenario == 'combined':
        return "🔴 GNSS unavailable + IMU disturbance — degraded adaptive mode"
    else:
        gnss_c = rep_state['gnss_conf']
        if gnss_c < 0.1:
            return "🔴 GNSS lost — Dead reckoning active"
        elif gnss_c < 0.5:
            return "🟡 GNSS degraded — Adaptive fusion active"
        return "🟢 All sensors healthy"


# ============================================================
# HELPER: Create Folium Map (scenario-aware)
# ============================================================

def create_map(df, meta, adaptive_result, baseline_traj,
               blackout_start=999, blackout_end=999, height=400):
    """
    Create a Folium map with all trajectories.
    GNSS Raw is filtered to stop during blackout windows.
    """
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

    # GNSS raw — filter out during blackout
    gnss_mask = ~df['gnss_lat'].isna()
    if blackout_start < 900:
        time_mask = ~((df['timestamp'] >= blackout_start) & (df['timestamp'] <= blackout_end))
        gnss_mask = gnss_mask & time_mask

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

    # Simulation banner — always visible
    st.markdown("""
    <div style="text-align:center; margin:-8px 0 12px;">
        <span style="display:inline-block; background:#fef7e0; color:#e37400;
                     padding:4px 14px; border-radius:12px; font-size:10px;
                     font-weight:700; letter-spacing:0.5px; text-transform:uppercase;
                     border:1px solid #fde293;">
            ⚠ Simulated Data — For Demonstration
        </span>
    </div>
    """, unsafe_allow_html=True)

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
    # GLOBAL SCENARIO STATE (persists across all tabs)
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

    # ============================================================
    # COMPUTE METRICS
    # ============================================================
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

    # ============================================================
    # REPRESENTATIVE STATE (scenario-aware, NOT just final timestep)
    # ============================================================
    rep = compute_representative_state(
        adaptive_result, scenario, blackout_start, blackout_end,
        gnss_anomaly_time, timestamps
    )

    # Representative values for display across ALL tabs
    rep_gnss_conf = rep['gnss_conf']
    rep_imu_conf = rep['imu_conf']
    rep_overall_conf = rep['overall_conf']
    rep_mode = rep['mode']

    # Compute GNSS status string
    if rep_gnss_conf < 0.05:
        gnss_status = "LOST"
    elif rep_gnss_conf < 0.3:
        gnss_status = "DEGRADED"
    elif rep_gnss_conf < 0.7:
        gnss_status = "REDUCED"
    else:
        gnss_status = "HEALTHY"

    # Build scenario-aware events
    road_summary = adaptive_result['road_detections']
    scenario_events = build_scenario_events(scenario, rep, road_summary)

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
    # 6-TAB LAYOUT
    # ============================================================
    tab_home, tab_sim, tab_nav, tab_test, tab_insights, tab_tech = st.tabs([
        "🏠 Home",
        "🚗 Simulator",
        "🧭 Navigate",
        "🧪 Test Lab",
        "📊 Insights",
        "🔬 Technical",
    ])

    # ── TAB 1: HOME ─────────────────────────────────────────
    with tab_home:
        # Active scenario badge
        st.markdown(f"""
        <div style="text-align:center; margin:4px 0 10px;">
            <span style="display:inline-block; background:#e8f0fe; color:#1a73e8;
                         padding:4px 14px; border-radius:12px; font-size:11px;
                         font-weight:700; border:1px solid #d2e3fc;">
                Active: {scenario_label}
            </span>
        </div>
        """, unsafe_allow_html=True)

        # Map (scenario-aware: GNSS raw filtered during blackout)
        m, _, _ = create_map(df, meta, adaptive_result, baseline_traj,
                             blackout_start=blackout_start,
                             blackout_end=blackout_end, height=380)
        st_folium(m, width=700, height=380, returned_objects=[], key="map_home")

        # Navigation Status — uses REPRESENTATIVE state
        render_section_title("Navigation Status")
        render_mode_indicator(rep_mode)

        # Scenario status text
        status_text = get_scenario_status_text(scenario, rep)
        st.markdown(f"""
        <div style="text-align:center; color:#5f6368; font-size:12px; margin:-4px 0 8px;">
            {status_text}
        </div>
        """, unsafe_allow_html=True)

        # Sensor Trust — LIVE from representative state
        render_section_title("Sensor Trust")
        col1, col2 = st.columns(2)
        with col1:
            gnss_label = f"GNSS ({gnss_status})"
            render_confidence_bar(gnss_label, rep_gnss_conf)
        with col2:
            imu_label = "IMU"
            if rep_imu_conf < 0.8:
                imu_label = "IMU (REDUCED)"
            render_confidence_bar(imu_label, rep_imu_conf)

        # Quick Stats — from actual pipeline metrics (these DO change per scenario)
        render_section_title("Quick Status")
        col1, col2, col3 = st.columns(3)
        with col1:
            render_key_metric(
                "Accuracy",
                f"{final_error:.1f} m",
                "pass" if final_error < 10 else "fail"
            )
        with col2:
            drift_val = adaptive_metrics.get('drift_percent', 0)
            render_key_metric("Drift", f"{drift_val:.2f}%",
                              "pass" if drift_val < 10 else "fail")
        with col3:
            render_key_metric(
                "Status",
                adaptive_metrics.get('pass_fail', 'N/A'),
                "pass" if adaptive_metrics.get('pass_fail') == 'PASS' else "fail"
            )

        # Navigation Mode during event
        st.markdown(f"""
        <div style="text-align:center; margin-top:8px; color:#80868b; font-size:11px;">
            Mode: <strong>{rep_mode}</strong> &nbsp;|&nbsp;
            GNSS: <strong style="color:{'#ea4335' if gnss_status != 'HEALTHY' else '#34a853'};">{gnss_status}</strong> &nbsp;|&nbsp;
            RMSE: <strong>{adaptive_metrics.get('rmse_m', 0):.1f}m</strong>
        </div>
        """, unsafe_allow_html=True)

    # ── TAB: SIMULATOR ───────────────────────────────────────
    with tab_sim:
        st.markdown("""
        <div style="text-align:center; margin:0 0 8px;">
            <span style="display:inline-block; background:#fef7e0; color:#e37400;
                         padding:4px 14px; border-radius:12px; font-size:10px;
                         font-weight:700; letter-spacing:0.5px; text-transform:uppercase;
                         border:1px solid #fde293;">
                Simulation — Synthetic Data
            </span>
        </div>
        """, unsafe_allow_html=True)

        render_section_title("🚗 GNSS Failure Simulator")

        st.markdown("""
        <div style="color:#5f6368; font-size:13px; margin-bottom:12px; line-height:1.5;">
            Watch SAGE handle real-time GNSS failures. Select an environment,
            press START, and observe how the system detects failures, switches
            to dead reckoning, and recovers when GNSS returns.<br>
            <em>Short-term GNSS-denied positioning using inertial dead reckoning.</em>
        </div>
        """, unsafe_allow_html=True)

        # ── Controls ────────────────────────────────────────
        col_env, col_sage = st.columns([2, 1])
        with col_env:
            env_options = {k: f"{v['icon']} {v['name']}" for k, v in ENVIRONMENTS.items()}
            selected_env = st.selectbox(
                "Environment",
                options=list(env_options.keys()),
                format_func=lambda k: env_options[k],
                index=1,  # default to Tunnel
                key="sim_env",
            )
        with col_sage:
            sage_on = st.toggle("SAGE ON", value=True, key="sim_sage_on")

        env_info = ENVIRONMENTS[selected_env]
        st.markdown(f"""
        <div style="background:#f8f9fa; border-radius:8px; padding:10px 14px;
                    margin:4px 0 10px; font-size:12px; color:#5f6368;
                    border:1px solid #e5e7eb;">
            <strong>{env_info['icon']} {env_info['name']}:</strong> {env_info['description']}
        </div>
        """, unsafe_allow_html=True)

        # ── Initialize simulator ────────────────────────────
        # Cache key: if environment changes, re-run
        sim_cache_key = f"sim_result_{selected_env}"
        if sim_cache_key not in st.session_state:
            sim = GNSSFailureSimulator(df, meta, environment=selected_env)
            sim.run(
                run_adaptive_fn=run_adaptive_pipeline,
                run_baseline_fn=run_baseline_dr,
                compute_metrics_fn=compute_full_metrics,
            )
            st.session_state[sim_cache_key] = sim
        else:
            sim = st.session_state[sim_cache_key]

        max_frame = sim.n_frames - 1

        # ── Playback state ──────────────────────────────────
        if 'sim_frame' not in st.session_state:
            st.session_state.sim_frame = 0
        if 'sim_playing' not in st.session_state:
            st.session_state.sim_playing = False

        current_frame = st.session_state.sim_frame
        frame = sim.get_frame(current_frame)

        # ── Map ─────────────────────────────────────────────
        traj_data = sim.get_trajectory_coords_up_to(current_frame)

        sim_map = folium.Map(
            location=[frame['gt_lat'], frame['gt_lon']],
            zoom_start=16,
            tiles='CartoDB positron',
            attr='© CartoDB'
        )

        # Ground truth (full route, dimmed)
        if len(traj_data['gt']) > 1:
            folium.PolyLine(
                traj_data['gt'], color='#5f6368', weight=3,
                opacity=0.3, dash_array='6', tooltip='Ground Truth'
            ).add_to(sim_map)

        # GNSS raw (only when available — stops during blackout)
        if len(traj_data['gnss']) > 1:
            folium.PolyLine(
                traj_data['gnss'], color='#fbbc04', weight=2,
                opacity=0.5, dash_array='4', tooltip='GNSS Raw'
            ).add_to(sim_map)

        if sage_on:
            # SAGE trajectory
            if len(traj_data['sage']) > 1:
                folium.PolyLine(
                    traj_data['sage'], color='#34a853', weight=3,
                    opacity=0.9, tooltip='SAGE Adaptive'
                ).add_to(sim_map)

            # Current position = SAGE position
            folium.Marker(
                [frame['sage_lat'], frame['sage_lon']],
                icon=folium.Icon(color='green', icon='location-dot', prefix='fa'),
                tooltip=f'SAGE Position (t={frame["time"]:.0f}s)',
            ).add_to(sim_map)
        else:
            # Baseline INS trajectory (drifts)
            if len(traj_data['baseline']) > 1:
                folium.PolyLine(
                    traj_data['baseline'], color='#ea4335', weight=2,
                    opacity=0.6, dash_array='8', tooltip='Baseline INS (drift)'
                ).add_to(sim_map)

            # Current position = baseline (shows the drift problem)
            folium.Marker(
                [frame['baseline_lat'], frame['baseline_lon']],
                icon=folium.Icon(color='red', icon='location-dot', prefix='fa'),
                tooltip=f'INS Position (t={frame["time"]:.0f}s) — No SAGE',
            ).add_to(sim_map)

        # Start marker
        folium.Marker(
            [sim.frames[0]['gt_lat'], sim.frames[0]['gt_lon']],
            icon=folium.Icon(color='blue', icon='flag', prefix='fa'),
            tooltip='Start',
        ).add_to(sim_map)

        # Legend
        sage_label = '<span style="color:#34a853;">━━</span> <b>SAGE Adaptive</b>' if sage_on else '<span style="color:#ea4335;">╍╍</span> Baseline INS'
        legend_html = f"""
        <div style="position:fixed; bottom:30px; left:10px; z-index:1000;
             background:white; padding:10px 14px; border-radius:8px;
             font-size:11px; color:#3c4043; box-shadow:0 1px 4px rgba(0,0,0,0.15);
             border:1px solid #e5e7eb; font-family:Inter,sans-serif;">
        <b style="font-size:12px;">t = {frame['time']:.0f}s</b><br>
        <span style="color:#5f6368;">╍╍</span> Ground Truth<br>
        <span style="color:#fbbc04;">╍╍</span> GNSS Raw<br>
        {sage_label}
        </div>
        """
        sim_map.get_root().html.add_child(folium.Element(legend_html))

        st_folium(sim_map, width=700, height=400, returned_objects=[], key="map_sim")

        # ── Time slider ─────────────────────────────────────
        slider_time = st.slider(
            "Time (seconds)",
            min_value=0.0,
            max_value=float(sim.frame_times[-1]),
            value=float(frame['time']),
            step=1.0,
            key="sim_slider",
            format="%.0fs",
        )

        # Sync slider → frame
        new_frame = int(np.argmin(np.abs(sim.frame_times - slider_time)))
        if new_frame != current_frame and not st.session_state.sim_playing:
            st.session_state.sim_frame = new_frame
            st.rerun()

        # ── Playback buttons ────────────────────────────────
        col_start, col_pause, col_reset = st.columns(3)
        with col_start:
            if st.button("▶ START", key="sim_start", use_container_width=True):
                st.session_state.sim_playing = True
                if st.session_state.sim_frame >= max_frame:
                    st.session_state.sim_frame = 0
                st.rerun()
        with col_pause:
            if st.button("⏸ PAUSE", key="sim_pause", use_container_width=True):
                st.session_state.sim_playing = False
                st.rerun()
        with col_reset:
            if st.button("↺ RESET", key="sim_reset", use_container_width=True):
                st.session_state.sim_playing = False
                st.session_state.sim_frame = 0
                st.rerun()

        # ── Navigation Mode ─────────────────────────────────
        render_section_title("Navigation Mode")

        if sage_on:
            render_mode_indicator(frame['mode'])
        else:
            # Without SAGE: show what happens without adaptive navigation
            if not frame['gnss_available']:
                st.markdown("""
                <div class="mode-indicator mode-dr">
                    🔴 NAVIGATION UNRELIABLE<br>
                    <div style="font-size:11px; margin-top:4px; opacity:0.8;">
                        GNSS lost — No adaptive system — Position unknown
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="mode-indicator mode-gnss">
                    🟢 GNSS ONLY<br>
                    <div style="font-size:11px; margin-top:4px; opacity:0.8;">
                        No sensor fusion — Vulnerable to GNSS failure
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # SAGE ON/OFF status banner
        if sage_on:
            if frame['gnss_status'] == 'LOST':
                st.markdown("""
                <div class="alert-banner alert-gnss-lost">
                    🔴 GNSS SIGNAL LOST — SAGE ACTIVE — Dead reckoning via IMU
                </div>
                """, unsafe_allow_html=True)
            elif frame['gnss_status'] == 'VALIDATING':
                st.markdown("""
                <div class="alert-banner" style="background:#fef7e0; color:#e37400;
                     border:1px solid #fde293; border-radius:10px; padding:12px;
                     text-align:center; font-weight:600; font-size:13px;">
                    🟡 GNSS SIGNAL RESTORED — VALIDATING...
                </div>
                """, unsafe_allow_html=True)
            elif frame['gnss_status'] == 'RESTORED':
                st.markdown("""
                <div class="alert-banner alert-gnss-ok">
                    ✅ POSITION CORRECTED — GNSS validated, drift compensated
                </div>
                """, unsafe_allow_html=True)
            elif frame['gnss_status'] == 'DEGRADED':
                st.markdown("""
                <div class="alert-banner" style="background:#fef7e0; color:#e37400;
                     border:1px solid #fde293; border-radius:10px; padding:12px;
                     text-align:center; font-weight:600; font-size:13px;">
                    ⚠️ GNSS DEGRADED — Confidence reduced
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="alert-banner alert-gnss-ok">
                    🟢 All navigation sensors active — SAGE monitoring
                </div>
                """, unsafe_allow_html=True)
        else:
            if not frame['gnss_available']:
                st.markdown("""
                <div class="alert-banner alert-gnss-lost">
                    🔴 GNSS SIGNAL LOST — NO SAGE — Navigation failed
                </div>
                """, unsafe_allow_html=True)

        # ── Sensor Trust ────────────────────────────────────
        render_section_title("Sensor Trust")

        gnss_label = f"GNSS ({frame['gnss_status']})"
        imu_label = "IMU" + (" (REDUCED)" if frame['imu_conf'] < 0.8 else "")

        col1, col2 = st.columns(2)
        with col1:
            render_confidence_bar(gnss_label, frame['gnss_conf'])
        with col2:
            render_confidence_bar(imu_label, frame['imu_conf'])

        render_confidence_bar("Overall", frame['overall_conf'])

        # ── Position ────────────────────────────────────────
        render_section_title("Position")

        current_error = frame['sage_error'] if sage_on else frame['baseline_error']

        col1, col2, col3 = st.columns(3)
        with col1:
            render_key_metric(
                "Error",
                f"{current_error:.1f} m",
                "pass" if current_error < 10 else "fail"
            )
        with col2:
            render_key_metric("Speed", f"{frame['speed_kmh']:.0f} km/h")
        with col3:
            render_key_metric("Heading", f"{frame['heading_deg']:.0f}°")

        # ── Event Timeline ──────────────────────────────────
        render_section_title("Event Timeline")

        events = sim.get_events_up_to(frame['time'])
        for evt_time, evt_level, evt_msg, is_past in events:
            if is_past:
                if evt_level == 'ok':
                    css = "event-ok"
                elif evt_level == 'error':
                    css = "event-error"
                else:
                    css = "event-warn"
                st.markdown(
                    f'<div class="event-item {css}">'
                    f'<strong>{evt_time:.0f}s</strong> &nbsp; {evt_msg}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="event-item" style="opacity:0.35; border-left:4px solid #dadce0;">'
                    f'<strong>{evt_time:.0f}s</strong> &nbsp; {evt_msg}</div>',
                    unsafe_allow_html=True
                )

        # ── Auto-play logic ─────────────────────────────────
        if st.session_state.sim_playing:
            time.sleep(0.3)
            st.session_state.sim_frame = min(
                st.session_state.sim_frame + 1,
                max_frame
            )
            if st.session_state.sim_frame >= max_frame:
                st.session_state.sim_playing = False
            st.rerun()

    # ── TAB 2: LIVE NAVIGATION ───────────────────────────────
    with tab_nav:
        # Scenario badge
        st.markdown(f"""
        <div style="text-align:center; margin:4px 0 10px;">
            <span style="display:inline-block; background:#e8f0fe; color:#1a73e8;
                         padding:4px 14px; border-radius:12px; font-size:11px;
                         font-weight:700; border:1px solid #d2e3fc;">
                {scenario_label}
            </span>
        </div>
        """, unsafe_allow_html=True)

        # Large Map (scenario-aware GNSS filtering)
        m_nav, _, _ = create_map(df, meta, adaptive_result, baseline_traj,
                                 blackout_start=blackout_start,
                                 blackout_end=blackout_end, height=450)
        st_folium(m_nav, width=700, height=450, returned_objects=[], key="map_nav")

        # Live Position
        render_section_title("Live Position")
        render_live_position(final_lat, final_lon, final_speed,
                             final_heading, final_error)

        # Navigation Mode — REPRESENTATIVE state
        render_section_title("Navigation Mode")
        render_mode_indicator(rep_mode)

        # Scenario-aware GNSS alert
        if rep_gnss_conf < 0.05:
            st.markdown(
                '<div class="alert-banner alert-gnss-lost">'
                '🔴 GNSS SIGNAL LOST — Estimating position via IMU dead reckoning'
                '</div>',
                unsafe_allow_html=True
            )
        elif rep_gnss_conf < 0.5:
            st.markdown(
                '<div class="alert-banner alert-gnss-lost">'
                f'⚠️ GNSS DEGRADED — Confidence {rep_gnss_conf*100:.0f}%'
                '</div>',
                unsafe_allow_html=True
            )
        elif rep_imu_conf < 0.8 and scenario != 'normal':
            st.markdown(
                '<div class="alert-banner" style="background:#fef7e0; color:#e37400; '
                'border:1px solid #fde293; border-radius:10px; padding:12px; '
                'text-align:center; font-weight:600; font-size:13px;">'
                f'⚠️ IMU quality reduced — Confidence {rep_imu_conf*100:.0f}%'
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

        # Combined scenario special banner
        if scenario == 'combined':
            st.markdown(
                '<div class="alert-banner alert-gnss-lost">'
                '⚠️ GNSS LOST + ROAD DISTURBANCE — Degraded adaptive mode'
                '</div>',
                unsafe_allow_html=True
            )

        # Sensor Confidence — REPRESENTATIVE values
        render_section_title("Sensor Confidence")
        render_confidence_bar(f"GNSS ({gnss_status})", rep_gnss_conf)
        render_confidence_bar(
            f"IMU {'(REDUCED)' if rep_imu_conf < 0.8 else ''}",
            rep_imu_conf
        )
        render_confidence_bar("Overall", rep_overall_conf)

        # Events — scenario-aware
        render_events_panel(scenario_events)

    # ── TAB 3: TEST LAB ──────────────────────────────────────
    with tab_test:
        render_section_title("🧪 Select a Navigation Scenario")

        st.markdown("""
        <div style="color:#5f6368; font-size:13px; margin-bottom:12px;">
        Choose a failure scenario to stress-test SAGE's adaptive navigation.
        The selected scenario becomes the <strong>global active state</strong> —
        all tabs (Home, Navigate, Insights) will update to reflect this scenario.
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

        # ── Scenario Result Card ─────────────────────────────
        render_section_title("Scenario Result")

        # Prominent result card with scenario-specific values
        gnss_color = '#ea4335' if gnss_status != 'HEALTHY' else '#34a853'
        imu_color = '#e37400' if rep_imu_conf < 0.8 else '#34a853'

        st.markdown(f"""
        <div style="background:#ffffff; border-radius:14px; padding:20px;
                    margin:10px 0; border:2px solid {'#f5c6c4' if scenario != 'normal' else '#ceead6'};
                    box-shadow:0 2px 6px rgba(0,0,0,0.08);">
            <div style="text-align:center; font-size:11px; text-transform:uppercase;
                        letter-spacing:1.5px; color:#80868b; font-weight:700;
                        margin-bottom:12px;">
                Active Scenario
            </div>
            <div style="text-align:center; font-size:18px; font-weight:800;
                        color:#202124; margin-bottom:14px;">
                {scenario_label}
            </div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
                <div style="background:#f8f9fa; border-radius:8px; padding:10px 14px;">
                    <div style="font-size:10px; color:#80868b; text-transform:uppercase;
                                letter-spacing:1px; font-weight:600;">GNSS Status</div>
                    <div style="font-size:16px; font-weight:700; color:{gnss_color};">
                        {gnss_status}
                    </div>
                </div>
                <div style="background:#f8f9fa; border-radius:8px; padding:10px 14px;">
                    <div style="font-size:10px; color:#80868b; text-transform:uppercase;
                                letter-spacing:1px; font-weight:600;">GNSS Trust</div>
                    <div style="font-size:16px; font-weight:700; color:{gnss_color};">
                        {rep_gnss_conf*100:.0f}%
                    </div>
                </div>
                <div style="background:#f8f9fa; border-radius:8px; padding:10px 14px;">
                    <div style="font-size:10px; color:#80868b; text-transform:uppercase;
                                letter-spacing:1px; font-weight:600;">IMU Trust</div>
                    <div style="font-size:16px; font-weight:700; color:{imu_color};">
                        {rep_imu_conf*100:.0f}%
                    </div>
                </div>
                <div style="background:#f8f9fa; border-radius:8px; padding:10px 14px;">
                    <div style="font-size:10px; color:#80868b; text-transform:uppercase;
                                letter-spacing:1px; font-weight:600;">Nav Mode</div>
                    <div style="font-size:14px; font-weight:700; color:#3c4043;">
                        {rep_mode.split('(')[0].strip()}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Metric results
        col1, col2, col3 = st.columns(3)
        with col1:
            render_key_metric("Final Error", f"{final_error:.1f} m",
                              "pass" if final_error < 10 else "fail")
        with col2:
            render_key_metric("Drift", f"{adaptive_metrics.get('drift_percent', 0):.2f}%",
                              "pass" if adaptive_metrics.get('drift_percent', 0) < 10 else "fail")
        with col3:
            render_key_metric("RMSE", f"{adaptive_metrics.get('rmse_m', 0):.1f} m")

        # Road disturbance summary
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
        # Scenario badge
        st.markdown(f"""
        <div style="text-align:center; margin:0 0 10px;">
            <span style="display:inline-block; background:#e8f0fe; color:#1a73e8;
                         padding:4px 14px; border-radius:12px; font-size:11px;
                         font-weight:700; border:1px solid #d2e3fc;">
                {scenario_label}
            </span>
        </div>
        """, unsafe_allow_html=True)

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

        col5, col6 = st.columns(2)
        with col5:
            render_key_metric("Distance", f"{adaptive_metrics.get('distance_travelled_m', 0):.0f} m")
        with col6:
            render_key_metric("Nav Mode", rep_mode.split('(')[0].strip())

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
        # Scenario badge
        st.markdown(f"""
        <div style="text-align:center; margin:0 0 10px;">
            <span style="display:inline-block; background:#e8f0fe; color:#1a73e8;
                         padding:4px 14px; border-radius:12px; font-size:11px;
                         font-weight:700; border:1px solid #d2e3fc;">
                {scenario_label}
            </span>
        </div>
        """, unsafe_allow_html=True)

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
