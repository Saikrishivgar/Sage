"""
Dashboard Visualization Components
====================================
Styled components for the mobile-style Streamlit dashboard.
"""

import streamlit as st
import numpy as np
from typing import Dict, List


def apply_mobile_style():
    """Apply mobile-style CSS to Streamlit app."""
    st.markdown("""
    <style>
    /* Mobile-style container */
    .main .block-container {
        max-width: 480px !important;
        padding: 1rem 1.5rem !important;
        margin: 0 auto !important;
    }
    
    /* Dark theme override */
    .stApp {
        background-color: #0a0a1a !important;
    }
    
    /* Header bar */
    .nav-header {
        background: linear-gradient(135deg, #1a1a3e 0%, #0d0d2b 100%);
        border-radius: 16px;
        padding: 16px 20px;
        margin-bottom: 16px;
        border: 1px solid rgba(99, 102, 241, 0.3);
        text-align: center;
    }
    .nav-header h1 {
        color: #a5b4fc;
        font-size: 18px;
        margin: 0;
        font-weight: 700;
        letter-spacing: 1px;
    }
    .nav-header p {
        color: #6366f1;
        font-size: 11px;
        margin: 4px 0 0;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    /* Mode indicator */
    .mode-indicator {
        border-radius: 12px;
        padding: 10px 16px;
        margin: 8px 0;
        text-align: center;
        font-weight: 700;
        font-size: 14px;
        letter-spacing: 1px;
    }
    .mode-gnss {
        background: linear-gradient(135deg, #065f46 0%, #064e3b 100%);
        color: #6ee7b7;
        border: 1px solid rgba(110, 231, 183, 0.3);
    }
    .mode-dr {
        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
        color: #fca5a5;
        border: 1px solid rgba(252, 165, 165, 0.3);
    }
    .mode-adaptive {
        background: linear-gradient(135deg, #78350f 0%, #92400e 100%);
        color: #fcd34d;
        border: 1px solid rgba(252, 211, 77, 0.3);
    }
    
    /* Confidence bar */
    .confidence-container {
        background: #111827;
        border-radius: 12px;
        padding: 14px 16px;
        margin: 6px 0;
        border: 1px solid #1f2937;
    }
    .conf-label {
        color: #9ca3af;
        font-size: 12px;
        margin-bottom: 6px;
        display: flex;
        justify-content: space-between;
    }
    .conf-bar-bg {
        background: #1f2937;
        border-radius: 6px;
        height: 10px;
        overflow: hidden;
    }
    .conf-bar-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.3s ease;
    }
    
    /* Metric card */
    .metric-card {
        background: #111827;
        border-radius: 12px;
        padding: 12px 16px;
        margin: 4px 0;
        border: 1px solid #1f2937;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .metric-label {
        color: #9ca3af;
        font-size: 12px;
    }
    .metric-value {
        color: #f3f4f6;
        font-size: 16px;
        font-weight: 700;
    }
    .metric-value.pass { color: #6ee7b7; }
    .metric-value.fail { color: #fca5a5; }
    .metric-value.warn { color: #fcd34d; }
    
    /* Event log */
    .event-item {
        padding: 6px 12px;
        margin: 3px 0;
        border-radius: 8px;
        font-size: 13px;
        color: #d1d5db;
    }
    .event-ok { background: rgba(16, 185, 129, 0.1); border-left: 3px solid #10b981; }
    .event-warn { background: rgba(245, 158, 11, 0.1); border-left: 3px solid #f59e0b; }
    .event-error { background: rgba(239, 68, 68, 0.1); border-left: 3px solid #ef4444; }
    
    /* Section divider */
    .section-title {
        color: #6b7280;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin: 16px 0 8px;
        padding-bottom: 4px;
        border-bottom: 1px solid #1f2937;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Stress test buttons */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        padding: 8px 12px !important;
        width: 100% !important;
        border: 1px solid #374151 !important;
        background: #1f2937 !important;
        color: #e5e7eb !important;
    }
    .stButton > button:hover {
        background: #374151 !important;
        border-color: #6366f1 !important;
    }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    """Render the navigation header."""
    st.markdown("""
    <div class="nav-header">
        <h1>🧭 INTELLIGENT DEAD RECKONING</h1>
        <p>AI-ML Adaptive Navigation System</p>
    </div>
    """, unsafe_allow_html=True)


def render_mode_indicator(mode: str):
    """Render navigation mode indicator."""
    if "DEAD RECKONING" in mode:
        css_class = "mode-dr"
        icon = "🔴"
    elif "DEGRADED" in mode or "ADAPTIVE" in mode:
        css_class = "mode-adaptive"
        icon = "⚠️"
    else:
        css_class = "mode-gnss"
        icon = "🟢"
    
    st.markdown(f"""
    <div class="mode-indicator {css_class}">
        {icon} {mode}
    </div>
    """, unsafe_allow_html=True)


def render_confidence_bar(label: str, value: float, color: str = "#6366f1"):
    """Render a confidence bar."""
    pct = max(0, min(100, int(value * 100)))
    
    if value > 0.7:
        bar_color = "#10b981"
    elif value > 0.3:
        bar_color = "#f59e0b"
    else:
        bar_color = "#ef4444"
    
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


def render_event(message: str, level: str = "ok"):
    """Render an event log item."""
    css_class = {"ok": "event-ok", "warn": "event-warn", "error": "event-error"}.get(level, "event-ok")
    st.markdown(f'<div class="event-item {css_class}">{message}</div>', unsafe_allow_html=True)


def render_section_title(title: str):
    """Render a section divider."""
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def render_accuracy_panel(metrics: Dict):
    """Render the full accuracy panel."""
    render_section_title("📊 ACCURACY")
    
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


def render_events_panel(events: List[str]):
    """Render the events panel."""
    render_section_title("📋 EVENTS")
    for event in events:
        if "✅" in event or "✓" in event:
            render_event(event, "ok")
        elif "🔴" in event or "OUTAGE" in event:
            render_event(event, "error")
        elif "⚠️" in event:
            render_event(event, "warn")
        else:
            render_event(event, "ok")
