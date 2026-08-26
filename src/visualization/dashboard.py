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
    /* Premium Mobile-style container */
    .main .block-container {
        max-width: 600px !important;
        padding: 2rem !important;
        margin: 0 auto !important;
    }
    
    /* Light theme override */
    .stApp {
        background-color: #f8f9fa !important;
        color: #1f2937 !important;
    }
    
    /* Header bar */
    .nav-header {
        background: #ffffff;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        text-align: center;
        border: 1px solid #e5e7eb;
    }
    .nav-header h1 {
        color: #111827;
        font-size: 22px;
        margin: 0;
        font-weight: 800;
        letter-spacing: 0.5px;
    }
    .nav-header p {
        color: #6366f1;
        font-size: 12px;
        margin: 6px 0 0;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 1.5px;
    }
    
    /* Mode indicator */
    .mode-indicator {
        border-radius: 12px;
        padding: 12px 20px;
        margin: 12px 0;
        text-align: center;
        font-weight: 700;
        font-size: 15px;
        letter-spacing: 0.5px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }
    .mode-gnss {
        background: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }
    .mode-dr {
        background: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }
    .mode-adaptive {
        background: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
    }
    
    /* Confidence bar */
    .confidence-container {
        background: #ffffff;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 8px 0;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    .conf-label {
        color: #4b5563;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
    }
    .conf-bar-bg {
        background: #f3f4f6;
        border-radius: 8px;
        height: 12px;
        overflow: hidden;
    }
    .conf-bar-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 0.4s ease-out;
    }
    
    /* Metric card */
    .metric-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 14px 16px;
        margin: 6px 0;
        border: 1px solid #e5e7eb;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    .metric-label {
        color: #6b7280;
        font-size: 13px;
        font-weight: 500;
    }
    .metric-value {
        color: #111827;
        font-size: 16px;
        font-weight: 800;
    }
    .metric-value.pass { color: #10b981; }
    .metric-value.fail { color: #ef4444; }
    .metric-value.warn { color: #f59e0b; }
    
    /* Event log */
    .event-item {
        padding: 10px 14px;
        margin: 6px 0;
        border-radius: 8px;
        font-size: 14px;
        color: #374151;
        background: #ffffff;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    .event-ok { border-left: 4px solid #10b981; }
    .event-warn { border-left: 4px solid #f59e0b; }
    .event-error { border-left: 4px solid #ef4444; }
    
    /* Section divider */
    .section-title {
        color: #4b5563;
        font-size: 12px;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 1.5px;
        margin: 24px 0 12px;
        padding-bottom: 6px;
        border-bottom: 2px solid #e5e7eb;
    }
    
    /* Tabs styling for premium look */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #ffffff;
        border-radius: 12px;
        padding: 6px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 8px;
        color: #6b7280;
        font-weight: 600;
        padding: 0 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #f3f4f6 !important;
        color: #111827 !important;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Stress test buttons */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 10px 16px !important;
        width: 100% !important;
        border: 1px solid #e5e7eb !important;
        background: #ffffff !important;
        color: #374151 !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
    }
    .stButton > button:hover {
        background: #f9fafb !important;
        border-color: #6366f1 !important;
        color: #4f46e5 !important;
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
