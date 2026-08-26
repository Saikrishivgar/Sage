"""
GNSS Failure Simulator
========================
Animated demonstration of SAGE's GNSS failure handling.

Reuses the existing pipeline:
- generate_dataset.py for trajectory/sensor data
- run_adaptive_pipeline() for fusion
- run_baseline_dr() for baseline comparison

Each "environment" maps to specific pipeline parameters.
The simulator pre-computes everything, then exposes per-frame
data for animated playback.

SIMULATED DATA — FOR DEMONSTRATION
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from src.sensors.coordinate_transform import local_to_latlon


# ============================================================
# ENVIRONMENT DEFINITIONS
# ============================================================

ENVIRONMENTS = {
    'normal': {
        'name': 'Normal Road',
        'icon': '🛣️',
        'description': 'Clear road with full GNSS visibility',
        'blackout_start': 999,
        'blackout_end': 999,
        'disturbance_enabled': False,
        'gnss_anomaly_time': None,
        'event_zone_start': None,
        'event_zone_end': None,
        'zone_label': None,
    },
    'tunnel': {
        'name': 'Tunnel',
        'icon': '🚇',
        'description': 'Vehicle enters a tunnel — complete GNSS blockage for 25 seconds',
        'blackout_start': 30,
        'blackout_end': 55,
        'disturbance_enabled': False,
        'gnss_anomaly_time': None,
        'event_zone_start': 30,
        'event_zone_end': 55,
        'zone_label': 'TUNNEL',
    },
    'urban_canyon': {
        'name': 'Urban Canyon',
        'icon': '🏙️',
        'description': 'Tall buildings cause multipath — intermittent GNSS jumps',
        'blackout_start': 999,
        'blackout_end': 999,
        'disturbance_enabled': False,
        'gnss_anomaly_time': [35.0, 42.0, 50.0],
        'event_zone_start': 33,
        'event_zone_end': 52,
        'zone_label': 'URBAN CANYON',
    },
    'dense_forest': {
        'name': 'Dense Forest',
        'icon': '🌲',
        'description': 'Dense canopy blocks satellite signals for 20 seconds',
        'blackout_start': 35,
        'blackout_end': 55,
        'disturbance_enabled': False,
        'gnss_anomaly_time': None,
        'event_zone_start': 35,
        'event_zone_end': 55,
        'zone_label': 'DENSE FOREST',
    },
    'underground_parking': {
        'name': 'Underground Parking',
        'icon': '🅿️',
        'description': 'Vehicle enters underground structure — 40-second GNSS outage',
        'blackout_start': 25,
        'blackout_end': 65,
        'disturbance_enabled': False,
        'gnss_anomaly_time': None,
        'event_zone_start': 25,
        'event_zone_end': 65,
        'zone_label': 'UNDERGROUND',
    },
    'rough_road': {
        'name': 'Rough Road',
        'icon': '🚧',
        'description': 'Poor road surface causes IMU disturbance — GNSS remains available',
        'blackout_start': 999,
        'blackout_end': 999,
        'disturbance_enabled': True,
        'gnss_anomaly_time': None,
        'event_zone_start': 40,
        'event_zone_end': 50,
        'zone_label': 'ROUGH ROAD',
    },
}


class GNSSFailureSimulator:
    """
    Wraps the existing SAGE pipeline for animated frame-by-frame playback.

    Usage:
        sim = GNSSFailureSimulator(df, meta, environment='tunnel')
        sim.run(run_adaptive_fn, run_baseline_fn)
        frame = sim.get_frame(42.0)  # state at t=42s
    """

    def __init__(self, df: pd.DataFrame, meta: Dict, environment: str = 'tunnel'):
        self.df = df
        self.meta = meta
        self.env_key = environment
        self.env = ENVIRONMENTS.get(environment, ENVIRONMENTS['tunnel'])

        self.timestamps = df['timestamp'].values
        self.duration = self.timestamps[-1] - self.timestamps[0]
        self.dt_imu = 1.0 / meta.get('imu_rate', 100)
        self.lat0 = meta['lat0']
        self.lon0 = meta['lon0']

        # Will be populated by run()
        self.adaptive_result = None
        self.baseline_traj = None
        self.adaptive_metrics = None
        self.baseline_metrics = None

        # Downsampled frame data (1 Hz for playback)
        self.frames = None
        self.frame_times = None
        self.n_frames = 0

        # Pre-built event timeline
        self.event_timeline = []

    def run(self, run_adaptive_fn, run_baseline_fn, compute_metrics_fn):
        """
        Run the full pipeline using the existing functions.

        Parameters
        ----------
        run_adaptive_fn : callable
            The existing run_adaptive_pipeline() function.
        run_baseline_fn : callable
            The existing run_baseline_dr() function.
        compute_metrics_fn : callable
            The existing compute_full_metrics() function.
        """
        env = self.env

        # Run pipelines with environment-specific parameters
        self.baseline_traj = run_baseline_fn(self.df, self.meta)

        self.adaptive_result = run_adaptive_fn(
            self.df, self.meta,
            blackout_start=env['blackout_start'],
            blackout_end=env['blackout_end'],
            disturbance_enabled=env['disturbance_enabled'],
            gnss_anomaly_time=env['gnss_anomaly_time'],
        )

        # Compute metrics
        gt_x = self.df['gt_x'].values
        gt_y = self.df['gt_y'].values

        n_a = min(len(self.adaptive_result['x']), len(gt_x))
        self.adaptive_metrics = compute_metrics_fn(
            self.adaptive_result['x'][:n_a], self.adaptive_result['y'][:n_a],
            gt_x[:n_a], gt_y[:n_a],
            blackout_mask=self.adaptive_result['blackout_mask'][:n_a],
        )

        n_b = min(len(self.baseline_traj['x']), len(gt_x))
        self.baseline_metrics = compute_metrics_fn(
            self.baseline_traj['x'][:n_b], self.baseline_traj['y'][:n_b],
            gt_x[:n_b], gt_y[:n_b],
            blackout_mask=self.adaptive_result['blackout_mask'][:n_b],
        )

        # Build frames (downsample to 1 Hz)
        self._build_frames()

        # Build event timeline
        self._build_event_timeline()

    def _build_frames(self):
        """Downsample pipeline output to 1 Hz frames for playback."""
        imu_rate = self.meta.get('imu_rate', 100)
        step = imu_rate  # 1 frame per second

        gt_x = self.df['gt_x'].values
        gt_y = self.df['gt_y'].values
        gt_speed = self.df['gt_speed'].values
        gt_heading = self.df['gt_heading'].values

        # Ground truth lat/lon
        gt_lats = self.df['gt_lat'].values
        gt_lons = self.df['gt_lon'].values

        # SAGE lat/lon
        sage_lats, sage_lons = local_to_latlon(
            self.adaptive_result['x'], self.adaptive_result['y'],
            self.lat0, self.lon0
        )

        # Baseline lat/lon
        baseline_lats, baseline_lons = local_to_latlon(
            self.baseline_traj['x'], self.baseline_traj['y'],
            self.lat0, self.lon0
        )

        # GNSS raw lat/lon (from dataframe)
        gnss_lats = self.df['gnss_lat'].values
        gnss_lons = self.df['gnss_lon'].values

        # Confidence arrays (start from index 1)
        gnss_conf = self.adaptive_result['gnss_confidence']
        imu_conf = self.adaptive_result['imu_confidence']
        overall_conf = self.adaptive_result['overall_confidence']
        modes = self.adaptive_result['modes']

        # Position errors
        from src.evaluation.metrics import position_error
        sage_errors = position_error(
            self.adaptive_result['x'][:len(gt_x)],
            self.adaptive_result['y'][:len(gt_y)],
            gt_x, gt_y
        )
        baseline_errors = position_error(
            self.baseline_traj['x'][:len(gt_x)],
            self.baseline_traj['y'][:len(gt_y)],
            gt_x, gt_y
        )

        env = self.env
        blackout_start = env['blackout_start']
        blackout_end = env['blackout_end']

        frames = []
        frame_times = []

        n_total = len(self.timestamps)

        for frame_idx in range(0, n_total, step):
            t = self.timestamps[frame_idx]
            frame_times.append(t)

            # Confidence index (offset by 1 since conf arrays start at step 1)
            conf_idx = min(max(0, frame_idx - 1), len(gnss_conf) - 1)
            mode_idx = min(frame_idx, len(modes) - 1)

            # Is GNSS available at this time?
            in_blackout = (blackout_start <= t <= blackout_end) if blackout_start < 900 else False

            # GNSS status
            gc = float(gnss_conf[conf_idx]) if conf_idx < len(gnss_conf) else 1.0
            if gc < 0.05:
                gnss_status = 'LOST'
            elif gc < 0.3:
                gnss_status = 'VALIDATING'
            elif gc < 0.7:
                gnss_status = 'DEGRADED'
            else:
                gnss_status = 'AVAILABLE'

            # Override: just exited blackout and recovering
            if not in_blackout and blackout_start < 900:
                if blackout_end <= t < blackout_end + 3.0 and gc < 0.7:
                    gnss_status = 'VALIDATING'
                elif blackout_end + 3.0 <= t < blackout_end + 5.0 and gc < 0.9:
                    gnss_status = 'RESTORED'

            frame = {
                'time': float(t),
                'frame_idx': frame_idx,
                # Ground truth
                'gt_lat': float(gt_lats[frame_idx]),
                'gt_lon': float(gt_lons[frame_idx]),
                'gt_speed': float(gt_speed[frame_idx]),
                'gt_heading': float(gt_heading[frame_idx]),
                # SAGE adaptive
                'sage_lat': float(sage_lats[min(frame_idx, len(sage_lats)-1)]),
                'sage_lon': float(sage_lons[min(frame_idx, len(sage_lons)-1)]),
                # Baseline INS
                'baseline_lat': float(baseline_lats[min(frame_idx, len(baseline_lats)-1)]),
                'baseline_lon': float(baseline_lons[min(frame_idx, len(baseline_lons)-1)]),
                # GNSS raw
                'gnss_lat': float(gnss_lats[frame_idx]) if not pd.isna(gnss_lats[frame_idx]) else None,
                'gnss_lon': float(gnss_lons[frame_idx]) if not pd.isna(gnss_lons[frame_idx]) else None,
                'gnss_available': not in_blackout,
                # Confidence
                'gnss_conf': gc,
                'imu_conf': float(imu_conf[conf_idx]) if conf_idx < len(imu_conf) else 1.0,
                'overall_conf': float(overall_conf[conf_idx]) if conf_idx < len(overall_conf) else 1.0,
                # Mode
                'mode': modes[mode_idx],
                'gnss_status': gnss_status,
                # Errors
                'sage_error': float(sage_errors[min(frame_idx, len(sage_errors)-1)]),
                'baseline_error': float(baseline_errors[min(frame_idx, len(baseline_errors)-1)]),
                # Speed
                'speed_kmh': float(gt_speed[frame_idx]) * 3.6,
                'heading_deg': float(np.degrees(gt_heading[frame_idx])) % 360,
                # In event zone
                'in_blackout': in_blackout,
            }

            frames.append(frame)

        self.frames = frames
        self.frame_times = np.array(frame_times)
        self.n_frames = len(frames)

    def _build_event_timeline(self):
        """Build the deterministic event timeline based on environment."""
        env = self.env
        events = []

        events.append((0.0, 'ok', '✅ Navigation started — GNSS available'))
        events.append((5.0, 'ok', f'✅ Vehicle moving — {env["name"]} scenario'))

        if self.env_key == 'normal':
            events.append((45.0, 'ok', '✅ All sensors nominal'))
            events.append((90.0, 'ok', '✅ Journey complete — No failures'))

        elif self.env_key in ('tunnel', 'dense_forest', 'underground_parking'):
            bs = env['blackout_start']
            be = env['blackout_end']
            label = env['zone_label']

            events.append((bs, 'error', f'🔴 ENTER {label} — GNSS SIGNAL LOST'))
            events.append((bs + 0.5, 'error', '🔴 SAGE detects GNSS failure'))
            events.append((bs + 1.0, 'error', '🔴 Dead reckoning active'))
            events.append(((bs + be) / 2, 'warn', f'⚠️ {be - bs:.0f}s without GNSS — IMU integration'))
            events.append((be, 'warn', f'🟡 EXIT {label}'))
            events.append((be + 1.0, 'warn', '🟡 GNSS signal detected — VALIDATING...'))
            events.append((be + 3.0, 'ok', '✅ GNSS RESTORED — Signal validated'))
            events.append((be + 5.0, 'ok', '✅ POSITION CORRECTED — Drift compensated'))
            events.append((min(be + 8.0, 89.0), 'ok', '✅ Full GNSS + INS navigation resumed'))

        elif self.env_key == 'urban_canyon':
            times = env['gnss_anomaly_time']
            events.append((33.0, 'warn', '🟡 Entering urban canyon — tall buildings'))
            for i, at in enumerate(times):
                events.append((at, 'error', f'⚠️ GNSS anomaly #{i+1} — ~55m position jump'))
                events.append((at + 0.5, 'warn', '⚠️ SAGE rejects anomalous GNSS fix'))
                events.append((at + 2.0, 'ok', '✅ GNSS confidence recovering'))
            events.append((52.0, 'ok', '✅ Exiting urban canyon'))

        elif self.env_key == 'rough_road':
            events.append((40.0, 'warn', '⚠️ Road disturbance detected — vibration spikes'))
            events.append((42.0, 'warn', '⚠️ IMU confidence reduced — sensor weights adjusted'))
            events.append((45.0, 'warn', '⚠️ Sustained disturbance — SAGE adapting'))
            events.append((48.5, 'warn', '⚠️ Final disturbance spike'))
            events.append((50.0, 'ok', '✅ Road surface normal — IMU confidence recovering'))
            events.append((55.0, 'ok', '✅ IMU confidence restored'))

        # Sort by time
        events.sort(key=lambda x: x[0])
        self.event_timeline = events

    def get_frame(self, frame_number: int) -> Dict:
        """Get all display data for a given frame number."""
        if self.frames is None:
            raise RuntimeError("Call run() first")
        idx = max(0, min(frame_number, self.n_frames - 1))
        return self.frames[idx]

    def get_events_up_to(self, current_time: float) -> List[Tuple]:
        """
        Get events up to current time.
        Returns list of (time, level, message) tuples.
        Past events are marked, future events are dimmed.
        """
        result = []
        for t, level, msg in self.event_timeline:
            if t <= current_time:
                result.append((t, level, msg, True))   # past/current
            else:
                result.append((t, 'future', msg, False))  # future
        return result

    def get_trajectory_coords_up_to(self, frame_number: int) -> Dict:
        """
        Get trajectory coordinates up to the current frame for progressive map drawing.
        Returns dict with lists of (lat, lon) tuples for each trajectory.
        """
        if self.frames is None:
            raise RuntimeError("Call run() first")

        idx = max(0, min(frame_number, self.n_frames - 1))

        gt_coords = []
        sage_coords = []
        baseline_coords = []
        gnss_coords = []

        for i in range(idx + 1):
            f = self.frames[i]
            gt_coords.append((f['gt_lat'], f['gt_lon']))
            sage_coords.append((f['sage_lat'], f['sage_lon']))
            baseline_coords.append((f['baseline_lat'], f['baseline_lon']))
            if f['gnss_available'] and f['gnss_lat'] is not None:
                gnss_coords.append((f['gnss_lat'], f['gnss_lon']))

        return {
            'gt': gt_coords,
            'sage': sage_coords,
            'baseline': baseline_coords,
            'gnss': gnss_coords,
        }
