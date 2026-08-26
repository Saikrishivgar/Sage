"""
SAGE Dashboard Visualization Components
=========================================
Premium navigation-app styled components for the Streamlit dashboard.
White/light theme with cards, shadows, and clean typography.
"""

import streamlit as st
import numpy as np
from typing import Dict, List


def apply_sage_style():
    """Apply premium navigation-app CSS to Streamlit app."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    /* Global */
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }

    .main .block-container {
        max-width: 720px !important;
        padding: 1.5rem 2rem !important;
        margin: 0 auto !important;
    }

    .stApp {
        background-color: #f0f2f5 !important;
    }

    /* ── SAGE Header ────────────────────────────── */
    .sage-header {
        background: #ffffff;
        border-radius: 16px;
        padding: 28px 24px 20px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        text-align: center;
        border: 1px solid #e5e7eb;
    }
    .sage-header .sage-logo {
        font-size: 32px;
        font-weight: 900;
        color: #1a73e8;
        letter-spacing: -0.5px;
        margin: 0;
    }
    .sage-header .sage-subtitle {
        font-size: 11px;
        color: #5f6368;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 600;
        margin: 4px 0 0;
    }
    .sage-header .sage-tagline {
        font-size: 13px;
        color: #80868b;
        margin: 8px 0 0;
        font-style: italic;
    }

    /* ── Navigation Tabs ────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #ffffff;
        border-radius: 12px;
        padding: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border: 1px solid #e5e7eb;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 8px;
        color: #5f6368;
        font-weight: 600;
        font-size: 13px;
        padding: 0 14px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e8f0fe !important;
        color: #1a73e8 !important;
    }

    /* ── Cards ───────────────────────────────────── */
    .sage-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 8px 0;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 2px rgba(0,0,0,0.06);
    }
    .sage-card-title {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #80868b;
        font-weight: 700;
        margin-bottom: 8px;
    }

    /* ── Mode Indicator ──────────────────────────── */
    .mode-indicator {
        border-radius: 12px;
        padding: 14px 20px;
        margin: 10px 0;
        text-align: center;
        font-weight: 700;
        font-size: 15px;
        letter-spacing: 0.3px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .mode-gnss {
        background: #e6f4ea;
        color: #137333;
        border: 1px solid #ceead6;
    }
    .mode-dr {
        background: #fce8e6;
        color: #c5221f;
        border: 1px solid #f5c6c4;
    }
    .mode-adaptive {
        background: #fef7e0;
        color: #e37400;
        border: 1px solid #fde293;
    }
    .mode-recovering {
        background: #e8f0fe;
        color: #1a73e8;
        border: 1px solid #d2e3fc;
    }

    /* ── Confidence Bars ─────────────────────────── */
    .confidence-container {
        background: #ffffff;
        border-radius: 12px;
        padding: 14px 18px;
        margin: 6px 0;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .conf-label {
        color: #3c4043;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 6px;
        display: flex;
        justify-content: space-between;
    }
    .conf-bar-bg {
        background: #f1f3f4;
        border-radius: 6px;
        height: 10px;
        overflow: hidden;
    }
    .conf-bar-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.4s ease-out;
    }

    /* ── Metric Cards ────────────────────────────── */
    .metric-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 14px 18px;
        margin: 5px 0;
        border: 1px solid #e5e7eb;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .metric-label {
        color: #5f6368;
        font-size: 13px;
        font-weight: 500;
    }
    .metric-value {
        color: #202124;
        font-size: 16px;
        font-weight: 700;
    }
    .metric-value.pass { color: #137333; }
    .metric-value.fail { color: #c5221f; }
    .metric-value.warn { color: #e37400; }

    /* ── Key Metric (large) ──────────────────────── */
    .key-metric {
        background: #ffffff;
        border-radius: 14px;
        padding: 18px 20px;
        margin: 6px 0;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        text-align: center;
    }
    .key-metric .km-value {
        font-size: 28px;
        font-weight: 800;
        color: #202124;
        line-height: 1.2;
    }
    .key-metric .km-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #80868b;
        font-weight: 600;
        margin-top: 4px;
    }
    .key-metric .km-value.pass { color: #137333; }
    .key-metric .km-value.fail { color: #c5221f; }

    /* ── Event Log ───────────────────────────────── */
    .event-item {
        padding: 10px 14px;
        margin: 5px 0;
        border-radius: 8px;
        font-size: 13px;
        color: #3c4043;
        background: #ffffff;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .event-ok { border-left: 4px solid #34a853; }
    .event-warn { border-left: 4px solid #fbbc04; }
    .event-error { border-left: 4px solid #ea4335; }

    /* ── Section Titles ──────────────────────────── */
    .section-title {
        color: #5f6368;
        font-size: 11px;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 1.5px;
        margin: 20px 0 10px;
        padding-bottom: 6px;
        border-bottom: 2px solid #e5e7eb;
    }

    /* ── Scenario Cards ──────────────────────────── */
    .scenario-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 16px 18px;
        margin: 6px 0;
        border: 2px solid #e5e7eb;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .scenario-card:hover {
        border-color: #1a73e8;
        box-shadow: 0 2px 8px rgba(26,115,232,0.15);
    }
    .scenario-card.active {
        border-color: #1a73e8;
        background: #e8f0fe;
    }
    .scenario-title {
        font-size: 15px;
        font-weight: 700;
        color: #202124;
        margin-bottom: 4px;
    }
    .scenario-desc {
        font-size: 12px;
        color: #5f6368;
        line-height: 1.4;
    }

    /* ── Live Position Card ──────────────────────── */
    .live-pos-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
        margin: 10px 0;
    }
    .live-pos-item {
        background: #ffffff;
        border-radius: 10px;
        padding: 12px 14px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .live-pos-item .lp-label {
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #80868b;
        font-weight: 600;
    }
    .live-pos-item .lp-value {
        font-size: 18px;
        font-weight: 700;
        color: #202124;
        margin-top: 2px;
    }

    /* ── Pipeline Diagram ────────────────────────── */
    .pipeline-diagram {
        background: #ffffff;
        border-radius: 14px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        text-align: center;
    }
    .pipeline-step {
        display: inline-block;
        background: #e8f0fe;
        color: #1a73e8;
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        margin: 3px 2px;
    }
    .pipeline-arrow {
        display: inline-block;
        color: #80868b;
        font-size: 14px;
        margin: 0 2px;
    }

    /* ── Alert Banner ────────────────────────────── */
    .alert-banner {
        border-radius: 10px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 13px;
        font-weight: 600;
        text-align: center;
    }
    .alert-gnss-lost {
        background: #fce8e6;
        color: #c5221f;
        border: 1px solid #f5c6c4;
    }
    .alert-gnss-ok {
        background: #e6f4ea;
        color: #137333;
        border: 1px solid #ceead6;
    }
    .alert-sim {
        background: #fef7e0;
        color: #e37400;
        border: 1px solid #fde293;
        font-size: 11px;
        font-weight: 500;
    }

    /* ── Buttons ─────────────────────────────────── */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 10px 16px !important;
        width: 100% !important;
        border: 1px solid #dadce0 !important;
        background: #ffffff !important;
        color: #3c4043 !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button:hover {
        background: #f8f9fa !important;
        border-color: #1a73e8 !important;
        color: #1a73e8 !important;
        box-shadow: 0 1px 3px rgba(26,115,232,0.2) !important;
    }

    /* ── Simulation Badge ────────────────────────── */
    .sim-badge {
        display: inline-block;
        background: #fef7e0;
        color: #e37400;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    /* ── Footer ──────────────────────────────────── */
    .sage-footer {
        text-align: center;
        color: #80868b;
        font-size: 11px;
        padding: 16px 0 8px;
        border-top: 1px solid #e5e7eb;
        margin-top: 24px;
    }

    /* ── Hide Streamlit chrome ────────────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


def render_sage_header():
    """Render the SAGE branded header."""
    st.markdown("""
    <div class="sage-header">
        <div class="sage-logo">🧭 SAGE</div>
        <div class="sage-subtitle">Smart Adaptive Guidance Engine</div>
        <div class="sage-tagline">Navigation that keeps moving when GNSS doesn't.</div>
    </div>
    """, unsafe_allow_html=True)


def render_section_title(title: str):
    """Render a section divider."""
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def render_mode_indicator(mode: str):
    """Render navigation mode indicator with dramatic styling."""
    if "DEAD RECKONING" in mode:
        css_class = "mode-dr"
        icon = "🔴"
        extra = '<div style="font-size:11px; margin-top:4px; opacity:0.8;">GNSS SIGNAL LOST — Navigating via IMU</div>'
    elif "DEGRADED" in mode:
        css_class = "mode-adaptive"
        icon = "⚠️"
        extra = '<div style="font-size:11px; margin-top:4px; opacity:0.8;">Sensor quality reduced</div>'
    elif "ADAPTIVE" in mode:
        css_class = "mode-adaptive"
        icon = "🟠"
        extra = '<div style="font-size:11px; margin-top:4px; opacity:0.8;">Adjusting sensor weights</div>'
    else:
        css_class = "mode-gnss"
        icon = "🟢"
        extra = '<div style="font-size:11px; margin-top:4px; opacity:0.8;">All sensors healthy</div>'

    st.markdown(f"""
    <div class="mode-indicator {css_class}">
        {icon} {mode}
        {extra}
    </div>
    """, unsafe_allow_html=True)


def render_confidence_bar(label: str, value: float):
    """Render a confidence bar."""
    pct = max(0, min(100, int(value * 100)))

    if value > 0.7:
        bar_color = "#34a853"
    elif value > 0.3:
        bar_color = "#fbbc04"
    else:
        bar_color = "#ea4335"

    st.markdown(f"""
    <div class="confidence-container">
        <div class="conf-label">
            <span>{label}</span>
            <span style="color:{bar_color}; font-weight:700;">{pct}%</span>
        </div>
        <div class="conf-bar-bg">
            <div class="conf-bar-fill" style="width:{pct}%; background:{bar_color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_metric(label: str, value: str, status: str = "normal"):
    """Render a metric card."""
    css_class = {"pass": "pass", "fail": "fail", "warn": "warn"}.get(status, "")

    st.markdown(f"""
    <div class="metric-card">
        <span class="metric-label">{label}</span>
        <span class="metric-value {css_class}">{value}</span>
    </div>
    """, unsafe_allow_html=True)


def render_key_metric(label: str, value: str, status: str = "normal"):
    """Render a large key metric card."""
    css_class = {"pass": "pass", "fail": "fail"}.get(status, "")

    st.markdown(f"""
    <div class="key-metric">
        <div class="km-value {css_class}">{value}</div>
        <div class="km-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def render_event(message: str, level: str = "ok"):
    """Render an event log item."""
    css_class = {"ok": "event-ok", "warn": "event-warn", "error": "event-error"}.get(level, "event-ok")
    st.markdown(f'<div class="event-item {css_class}">{message}</div>', unsafe_allow_html=True)


def render_events_panel(events: List[str]):
    """Render the events panel."""
    render_section_title("Live Events")
    for event in events:
        if "✅" in event or "✓" in event:
            render_event(event, "ok")
        elif "🔴" in event or "OUTAGE" in event:
            render_event(event, "error")
        elif "⚠️" in event:
            render_event(event, "warn")
        else:
            render_event(event, "ok")


def render_live_position(lat: float, lon: float, speed_ms: float,
                         heading_rad: float, accuracy_m: float):
    """Render the live position card with lat/lon/speed/heading/accuracy."""
    speed_kmh = speed_ms * 3.6
    heading_deg = np.degrees(heading_rad) % 360

    st.markdown(f"""
    <div class="live-pos-grid">
        <div class="live-pos-item">
            <div class="lp-label">Latitude</div>
            <div class="lp-value">{lat:.6f}°</div>
        </div>
        <div class="live-pos-item">
            <div class="lp-label">Longitude</div>
            <div class="lp-value">{lon:.6f}°</div>
        </div>
        <div class="live-pos-item">
            <div class="lp-label">Speed</div>
            <div class="lp-value">{speed_kmh:.1f} <span style="font-size:12px;color:#80868b;">km/h</span></div>
        </div>
        <div class="live-pos-item">
            <div class="lp-label">Heading</div>
            <div class="lp-value">{heading_deg:.0f}°</div>
        </div>
        <div class="live-pos-item">
            <div class="lp-label">Accuracy</div>
            <div class="lp-value">{accuracy_m:.1f} <span style="font-size:12px;color:#80868b;">m</span></div>
        </div>
        <div class="live-pos-item">
            <div class="lp-label">Status</div>
            <div class="lp-value" style="color:{'#34a853' if accuracy_m < 10 else '#ea4335'};">{'● Active' if accuracy_m < 50 else '● Lost'}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_scenario_card(title: str, icon: str, description: str,
                         expected: str, is_active: bool = False):
    """Render a scenario selection card."""
    active_class = "active" if is_active else ""
    st.markdown(f"""
    <div class="scenario-card {active_class}">
        <div class="scenario-title">{icon} {title}</div>
        <div class="scenario-desc">{description}</div>
        <div class="scenario-desc" style="margin-top:4px; color:#80868b;">
            <strong>Expected:</strong> {expected}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_pipeline_diagram():
    """Render the system pipeline diagram."""
    st.markdown("""
    <div class="pipeline-diagram">
        <div class="sage-card-title">SAGE Sensor Pipeline</div>
        <div>
            <span class="pipeline-step">📱 IMU</span>
            <span class="pipeline-arrow">→</span>
            <span class="pipeline-step">🔧 Preprocessing</span>
            <span class="pipeline-arrow">→</span>
            <span class="pipeline-step">📊 Quality</span>
            <span class="pipeline-arrow">→</span>
            <span class="pipeline-step">📈 Confidence</span>
            <span class="pipeline-arrow">→</span>
            <span class="pipeline-step">🔀 Fusion</span>
            <span class="pipeline-arrow">→</span>
            <span class="pipeline-step">📍 Position</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sim_badge():
    """Render a simulation data badge."""
    st.markdown(
        '<div style="text-align:center; margin:8px 0;">'
        '<span class="sim-badge">⚠ Simulated Data — Hyderabad Route</span>'
        '</div>',
        unsafe_allow_html=True
    )


def render_accuracy_panel(metrics: Dict):
    """Render the full accuracy panel."""
    render_section_title("Position Accuracy")

    render_metric("Current Error", f"{metrics.get('final_error_m', 0):.1f} m",
                  "pass" if metrics.get('final_error_m', 999) < 10 else "fail")
    render_metric("Mean Error", f"{metrics.get('mean_error_m', 0):.1f} m")
    render_metric("RMSE", f"{metrics.get('rmse_m', 0):.1f} m")
    render_metric("Max Error", f"{metrics.get('max_error_m', 0):.1f} m")

    drift = metrics.get('drift_percent', 0)
    target = metrics.get('target_drift_percent', 10)
    drift_status = "pass" if drift < target else "fail"
    render_metric("Drift", f"{drift:.2f}%", drift_status)
    render_metric("Target", f"< {target}%")
    render_metric("Distance", f"{metrics.get('distance_travelled_m', 0):.0f} m")

    status = metrics.get('pass_fail', 'N/A')
    render_metric("Status", status, "pass" if status == "PASS" else "fail")


def render_footer():
    """Render the SAGE footer."""
    st.markdown("""
    <div class="sage-footer">
        SAGE — Smart Adaptive Guidance Engine<br>
        Adaptive GNSS–IMU Navigation | Prototype v1.0<br>
        <em>ML-based confidence prediction — planned for Phase 2</em>
    </div>
    """, unsafe_allow_html=True)
