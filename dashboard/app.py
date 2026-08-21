"""
Streamlit Dashboard — AgriErgo Ergonomics & Drudgery Assessment Platform.

Provides a rich visual UI for uploading field-work videos, monitoring analysis,
and visualizing all 11 output parameters, activity timelines, posture distributions,
and REBA ergonomic risk scores.
"""

import sys
import os
import tempfile
import time
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (
    STREAMLIT_PAGE_TITLE,
    STREAMLIT_PAGE_ICON,
    SUPPORTED_FORMATS,
    MAX_UPLOAD_SIZE_MB,
)
from agriergo.pipeline import AgriErgoPipeline
from agriergo.analytics.parameter_aggregator import WorkerReport


# ──────────────────────────────────────────────
# Page Setup & Styling
# ──────────────────────────────────────────────
st.set_page_config(
    page_title=STREAMLIT_PAGE_TITLE,
    page_icon=STREAMLIT_PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished aesthetics
st.markdown("""
<style>
    /* Metric Cards */
    .metric-card {
        background-color: #1E293B;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #3B82F6;
        margin-bottom: 12px;
    }
    .metric-title {
        color: #94A3B8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .metric-value {
        color: #F8FAFC;
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 4px;
    }
    
    /* Risk Badge Colors */
    .risk-negligible { background-color: #10B981; color: white; padding: 4px 12px; border-radius: 12px; font-weight: bold; }
    .risk-low { background-color: #84CC16; color: white; padding: 4px 12px; border-radius: 12px; font-weight: bold; }
    .risk-medium { background-color: #F59E0B; color: white; padding: 4px 12px; border-radius: 12px; font-weight: bold; }
    .risk-high { background-color: #EF4444; color: white; padding: 4px 12px; border-radius: 12px; font-weight: bold; }
    .risk-very-high { background-color: #881337; color: white; padding: 4px 12px; border-radius: 12px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Main Header
# ──────────────────────────────────────────────
st.title(f"{STREAMLIT_PAGE_ICON} AgriErgo")
st.caption("Video-Based Farm Worker Ergonomics & Drudgery Assessment Platform — Phase 1 Prototype")
st.markdown("---")


# ──────────────────────────────────────────────
# Sidebar - Configuration
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    sample_fps = st.slider(
        "Sampling FPS",
        min_value=1,
        max_value=15,
        value=5,
        help="Higher FPS improves precision but increases processing time."
    )
    
    st.markdown("---")
    st.markdown("### About Parameters")
    st.markdown("""
    Extracts 11 key ergonomic parameters:
    1. **Sitting duration**
    2. **Standing duration**
    3. **Bending duration**
    4. **Walking duration**
    5. **Load carried**
    6. **Repetitive motion frequency**
    7. **Trip count**
    8. **Tools/equipment used**
    9. **Posture & Joint angles**
    10. **Continuous work duration**
    11. **Rest duration & count**
    """)


# ──────────────────────────────────────────────
# Step 1: Upload Video
# ──────────────────────────────────────────────
@st.cache_resource
def get_pipeline():
    """Cache AgriErgoPipeline model instances in RAM for 0s instant startup."""
    return AgriErgoPipeline()

# Sidebar Configuration
st.sidebar.title("⚙️ Pipeline Configuration")
speed_mode = st.sidebar.radio(
    "⚡ Processing Speed Mode",
    ["⚡ Lightning Fast (<5s)", "⚖️ Balanced Fast (Default)", "🔬 High Precision Research"],
    index=1,
    help="Lightning Fast processes 10-30 min videos in 3-5s. Balanced Fast processes in 8-12s."
)

sample_fps = 5.0

uploaded_file = st.file_uploader(
    "Upload Field-Work Video",
    type=[fmt.replace('.', '') for fmt in SUPPORTED_FORMATS],
    help="Upload an MP4, AVI, or MOV video clip of agricultural field work."
)

if uploaded_file is not None:
    # Read bytes and save to temp file for OpenCV processing
    file_bytes = uploaded_file.getvalue()
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix)
    tfile.write(file_bytes)
    tfile.close()
    video_path = tfile.name

    col1, col2 = st.columns([2, 1])
    with col1:
        try:
            st.video(file_bytes)
        except Exception as e:
            st.info("📹 Video file loaded for pipeline processing (browser preview unavailable for this codec format).")
    with col2:
        st.success(f"**File uploaded:** {uploaded_file.name}")
        st.info(f"**Size:** {round(len(file_bytes) / (1024*1024), 2)} MB")
        
        process_btn = st.button("🚀 Analyze Video", type="primary", use_container_width=True)

    if process_btn:
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def update_progress(pct: float, msg: str):
            progress_bar.progress(pct)
            status_text.text(msg)

        pipeline = get_pipeline()
        
        with st.spinner(f"Processing video ({speed_mode})..."):
            try:
                result = pipeline.process(
                    video_path,
                    progress_callback=update_progress,
                    speed_mode=speed_mode,
                )
            except TypeError:
                result = pipeline.process(
                    video_path,
                    progress_callback=update_progress,
                )

        st.session_state["pipeline_result"] = result
        st.success(f"⚡ Analysis complete in {result.processing_time_seconds}s!")


# ──────────────────────────────────────────────
# Step 2: Display Results
# ──────────────────────────────────────────────
if "pipeline_result" in st.session_state:
    result = st.session_state["pipeline_result"]
    
    st.markdown("---")
    st.header("📊 Assessment Results")

    # Overview KPI Metrics
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric("Video Duration", f"{result.video_metadata.duration_seconds}s")
    with kpi2:
        st.metric("Physical Workers", result.workers_detected)
    with kpi3:
        st.metric("Peak Concurrent", getattr(result, 'peak_concurrent_workers', result.workers_detected))
    with kpi4:
        st.metric("Processed Frames", result.frames_processed)
    with kpi5:
        st.metric("Processing Time", f"{result.processing_time_seconds}s")

    if not result.worker_reports:
        st.warning("No workers detected with sufficient frame history.")
    else:
        # ── Multi-Worker Comparative Leaderboard ──
        if len(result.worker_reports) > 1:
            st.markdown("### 🏆 Multi-Worker Ergonomic & Drudgery Leaderboard")
            leaderboard_data = [
                {
                    "Worker ID": f"Worker #{w.worker_id}",
                    "Auto-Classified Task": getattr(w, 'classified_task', 'General Work') or 'General Work',
                    "REBA Risk": f"{w.reba_score or 'N/A'} ({w.reba_risk_level or 'N/A'})",
                    "RULA Risk": getattr(w, 'rula_score', 'N/A') or 'N/A',
                    "Drudgery Index (ADI)": f"{getattr(w, 'drudgery_index', 0.0)} / 100",
                    "L5/S1 Compression": f"{getattr(w, 'l5s1_compression_n', 0.0)} N",
                    "Dominant Posture": w.posture_summary.dominant_posture.value.title() if w.posture_summary else "N/A",
                }
                for w in sorted(result.worker_reports, key=lambda x: getattr(x, 'drudgery_index', 0.0) or 0.0, reverse=True)
            ]
            st.dataframe(pd.DataFrame(leaderboard_data), use_container_width=True, hide_index=True)
            st.markdown("---")

        # Worker Selection
        worker_ids = [w.worker_id for w in result.worker_reports]
        selected_worker_id = st.selectbox("Select Worker to View Detailed Assessment", worker_ids, format_func=lambda x: f"Worker #{x}")

        report: WorkerReport = next(w for w in result.worker_reports if w.worker_id == selected_worker_id)

        task_name = getattr(report, 'classified_task', 'General Agricultural Work') or 'General Agricultural Work'
        task_hazard = getattr(report, 'task_hazard_profile', '') or ''
        st.subheader(f"Detailed Analysis for Worker #{selected_worker_id} — 🌾 {task_name}")
        if task_hazard:
            st.caption(f"**Primary Ergonomic Hazard Profile:** {task_hazard}")

        # Worker Overview Cards (6 columns)
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            st.metric("Tracked Time", f"{report.total_tracked_time}s")
        with c2:
            st.metric("REBA Score (Body)", f"{report.reba_score or 'N/A'}")
        with c3:
            st.metric("RULA Score (Upper)", f"{getattr(report, 'rula_score', 'N/A') or 'N/A'}")
        with c4:
            adi_val = getattr(report, 'drudgery_index', 0.0) or 0.0
            drudgery_pct = getattr(report, 'drudgery_percentage', 0.0) or adi_val
            st.metric("Drudgery Index / Pct", f"{drudgery_pct}%", help=f"Agricultural Drudgery Index: {adi_val} / 100")
        with c5:
            l5s1_val = getattr(report, 'l5s1_compression_n', 0.0) or 0.0
            st.metric("L5/S1 Compression", f"{l5s1_val} N")
        with c6:
            st.metric("Arm Postural Risk", f"{getattr(report, 'arm_postural_risk', 'Low') or 'Low'}")

        # ISO 11226 Ergonomic Shift Hazard Callout Banner
        if getattr(report, 'iso_11226_violated', False):
            st.error(f"⚠️ **{getattr(report, 'iso_11226_message', 'ISO 11226 Ergonomic Violation')}**")

        # Drudgery Recommendations Callout
        recs = getattr(report, 'drudgery_recommendations', [])
        if recs:
            st.info(f"💡 **Ergonomic Intervention Recommendation:** {recs[0]}")

        # ── SECTION 1: Standardised 1-Hour Activity Analysis ──
        st.markdown("### ⏱️ Standardisation for 1-Hour Activity Baseline")
        st.caption("Extrapolates and standardizes sampled video data to a standardized **1-Hour (3600 seconds)** continuous activity baseline for scientific ergonomic comparison.")

        std = getattr(report, 'standardised_1hr', None)
        if std:
            std_col1, std_col2, std_col3, std_col4 = st.columns(4)
            with std_col1:
                st.metric("Standard 1-Hr Duration", "01:00:00 (3600s)")
            with std_col2:
                st.metric("Standard Active Work", f"{std.active_work_formatted_1hr}", help="Total active work duration per 1-hour shift excluding pauses.")
            with std_col3:
                st.metric("Standard Rest / Pauses", f"{std.rest_formatted_1hr} ({std.rest_pct}%)", help="Standardised rest duration per 1-hour.")
            with std_col4:
                st.metric("Standard Repetitions (1-Hr)", f"{std.repetitive_cycles_1hr} cycles", help="Standardised repetitive work cycles performed per 1-hour.")

            # Standardised Durations Table
            std_table_data = [
                {"Action / Posture": "1. Sitting", "Sampled Tracked Time": f"{report.sitting_duration}s", "1-Hour Standardised Duration": std.sitting_formatted_1hr, "Share (%)": f"{std.sitting_pct}%", "State Entry Reps": getattr(report, 'sitting_reps', 0)},
                {"Action / Posture": "1b. Squatting", "Sampled Tracked Time": f"{getattr(report, 'squatting_duration', 0.0)}s", "1-Hour Standardised Duration": std.squatting_formatted_1hr, "Share (%)": f"{std.squatting_pct}%", "State Entry Reps": getattr(report, 'squatting_reps', 0)},
                {"Action / Posture": "2. Standing", "Sampled Tracked Time": f"{report.standing_duration}s", "1-Hour Standardised Duration": std.standing_formatted_1hr, "Share (%)": f"{std.standing_pct}%", "State Entry Reps": getattr(report, 'standing_reps', 0)},
                {"Action / Posture": "3. Bending (Stooping)", "Sampled Tracked Time": f"{report.bending_duration}s", "1-Hour Standardised Duration": std.bending_formatted_1hr, "Share (%)": f"{std.bending_pct}%", "State Entry Reps": getattr(report, 'bending_reps', 0)},
                {"Action / Posture": "4. Walking", "Sampled Tracked Time": f"{report.walking_duration}s", "1-Hour Standardised Duration": std.walking_formatted_1hr, "Share (%)": f"{std.walking_pct}%", "State Entry Reps": getattr(report, 'walking_reps', 0)},
                {"Action / Posture": "5. Carried Load Events", "Sampled Tracked Time": f"{report.total_load_events} events", "1-Hour Standardised Duration": f"{std.load_events_1hr} events/hr", "Share (%)": "N/A", "State Entry Reps": report.total_load_events},
                {"Action / Posture": "7. Work Pause (Rest)", "Sampled Tracked Time": f"{report.total_rest_duration}s", "1-Hour Standardised Duration": std.rest_formatted_1hr, "Share (%)": f"{std.rest_pct}%", "State Entry Reps": f"{std.rest_count_1hr} pauses/hr"},
            ]
            st.dataframe(pd.DataFrame(std_table_data), use_container_width=True, hide_index=True)
            st.markdown("---")

        # ── SECTION 2: Angle of Arms (Postural Study) ──
        st.markdown("### 💪 Angle of Arms — Postural Study (Shoulder Elevation & Elbow Flexion)")
        st.caption("Biomechanical vector tracking between **Shoulder → Elbow** and **Elbow → Wrist** to quantify upper arm elevation relative to torso and identify static overhead hazards.")

        ps = report.posture_summary
        avg_sh = getattr(ps, 'avg_shoulder_angle', None) if ps else None
        max_sh = getattr(ps, 'max_shoulder_angle', None) if ps else None
        avg_el = getattr(ps, 'avg_elbow_angle', None) if ps else None
        sh_45 = getattr(ps, 'shoulder_above_45_pct', 0.0) if ps else 0.0
        sh_90 = getattr(ps, 'shoulder_above_90_pct', 0.0) if ps else 0.0

        arm_c1, arm_c2, arm_c3, arm_c4 = st.columns(4)
        with arm_c1:
            st.metric("💪 Avg Shoulder Elevation", f"{avg_sh}°" if avg_sh is not None else "N/A", help="Upper arm angle relative to neutral torso axis.")
        with arm_c2:
            st.metric("⚡ Peak Shoulder Angle", f"{max_sh}°" if max_sh is not None else "N/A", help="Maximum arm elevation reach during activity.")
        with arm_c3:
            st.metric("📐 Avg Elbow Flexion", f"{avg_el}°" if avg_el is not None else "N/A", help="Elbow joint angle formed by upper arm and forearm.")
        with arm_c4:
            st.metric("Arms Elevated (>45°)", f"{sh_45}% ({report.shoulder_above_45_duration}s)", help="Percentage and duration of work spent with arms elevated >45°.")

        # Arm Angle Visualizations (Gauge + Multi-Line Timeline)
        arm_g1, arm_g2 = st.columns([1, 2])
        with arm_g1:
            st.markdown("#### 🧭 Shoulder Elevation Gauge")
            fig_arm_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=float(avg_sh or 0.0),
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Avg Upper Arm Elevation (°)", 'font': {'size': 15}},
                gauge={
                    'axis': {'range': [0, 140], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "#8B5CF6", 'thickness': 0.25},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 2,
                    'bordercolor': "#334155",
                    'steps': [
                        {'range': [0, 20], 'color': "rgba(16, 185, 129, 0.4)"},   # Safe Green (<20°)
                        {'range': [20, 45], 'color': "rgba(245, 158, 11, 0.4)"},  # Moderate Yellow (20°-45°)
                        {'range': [45, 90], 'color': "rgba(239, 68, 68, 0.5)"},   # Severe Red (45°-90°)
                        {'range': [90, 140], 'color': "rgba(217, 70, 239, 0.6)"}, # Overhead Violet (>90°)
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 45.0
                    }
                }
            ))
            fig_arm_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_arm_gauge, use_container_width=True)

        with arm_g2:
            st.markdown("#### 📈 Continuous Arm & Elbow Angle Curve Over Time")
            arm_time_series = getattr(ps, 'arm_angle_time_series', []) if ps else []
            if arm_time_series:
                arm_df = pd.DataFrame(arm_time_series)
                fig_arm_curve = px.line(
                    arm_df,
                    x="timestamp",
                    y=["shoulder_angle", "elbow_angle"],
                    labels={"timestamp": "Time (Seconds)", "value": "Joint Angle (°)", "variable": "Angle Metric"},
                    color_discrete_map={"shoulder_angle": "#8B5CF6", "elbow_angle": "#38BDF8"}
                )
                fig_arm_curve.add_hline(y=45, line_dash="dash", line_color="#EF4444", annotation_text="Arm Strain Limit (45°)")
                fig_arm_curve.add_hline(y=90, line_dash="dash", line_color="#D946EF", annotation_text="Overhead Work (90°)")
                fig_arm_curve.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20), legend_title_text="")
                st.plotly_chart(fig_arm_curve, use_container_width=True)
            else:
                st.info("Arm posture time-series recorded across video frames.")

        st.markdown("---")

        # ── SECTION 3: Dedicated Postural Angle & Plantation Stooping Assessment ──
        st.markdown("### 📐 Farmer Bending & Postural Angle Assessment (Plantation / Field Work)")
        
        avg_flex = report.posture_summary.avg_trunk_flexion if report.posture_summary else 0.0
        max_flex = report.posture_summary.max_trunk_flexion if report.posture_summary else 0.0
        avg_knee = getattr(report.posture_summary, 'avg_knee_angle', None) if report.posture_summary else None
        avg_hip = report.posture_summary.avg_hip_angle if report.posture_summary else 0.0

        # Angle Category
        if avg_flex is not None and avg_flex > 60.0:
            angle_cat = "🔴 Severe Bending / Stooping (High Hazard)"
        elif avg_flex is not None and avg_flex > 20.0:
            angle_cat = "🟡 Moderate Stooping (Monitor Exposure)"
        else:
            angle_cat = "🟢 Safe Upright / Neutral Posture"

        ang_col1, ang_col2, ang_col3, ang_col4 = st.columns(4)
        with ang_col1:
            st.metric("📏 Average Bending Angle", f"{avg_flex}°" if avg_flex is not None else "N/A", help="Average forward torso angle relative to vertical.")
        with ang_col2:
            st.metric("⚡ Peak Bending Angle", f"{max_flex}°" if max_flex is not None else "N/A", help="Maximum stoop angle reached during field work.")
        with ang_col3:
            st.metric("🦵 Average Knee Angle", f"{avg_knee}°" if avg_knee is not None else "N/A", help="Straight legs (>140°) indicate bending/stooping; bent legs (<90°) indicate squatting.")
        with ang_col4:
            st.metric("Severe Stoop Duration (>60°)", f"{report.severe_bending_duration}s", help="Cumulative time spent in severe trunk flexion >60°.")

        st.caption(f"**Bending Ergonomics Assessment:** {angle_cat}")

        # Interactive Angle Gauge & Continuous Bending Timeline
        angle_series = getattr(report.posture_summary, 'angle_time_series', []) if report.posture_summary else []
        g_col1, g_col2 = st.columns([1, 2])

        with g_col1:
            st.markdown("#### 🧭 Trunk Angle Gauge")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=float(avg_flex or 0.0),
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Avg Trunk Bending (°)", 'font': {'size': 16}},
                gauge={
                    'axis': {'range': [0, 120], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "#3B82F6", 'thickness': 0.25},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 2,
                    'bordercolor': "#334155",
                    'steps': [
                        {'range': [0, 20], 'color': "rgba(16, 185, 129, 0.4)"},   # Safe Green
                        {'range': [20, 60], 'color': "rgba(245, 158, 11, 0.4)"},  # Moderate Yellow
                        {'range': [60, 120], 'color': "rgba(239, 68, 68, 0.5)"},  # Severe Red
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 60.0
                    }
                }
            ))
            fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_gauge, use_container_width=True)

        with g_col2:
            st.markdown("#### 📈 Continuous Bending Angle Curve Over Time")
            if angle_series:
                angle_df = pd.DataFrame(angle_series)
                fig_angle_curve = px.line(
                    angle_df,
                    x="timestamp",
                    y="trunk_flexion",
                    labels={"timestamp": "Time (Seconds)", "trunk_flexion": "Trunk Flexion Angle (°)"},
                    color_discrete_sequence=["#F59E0B"]
                )
                fig_angle_curve.add_hline(y=20, line_dash="dash", line_color="#10B981", annotation_text="Safe Upright (20°)")
                fig_angle_curve.add_hline(y=60, line_dash="dash", line_color="#EF4444", annotation_text="Severe Stoop Limit (60°)")
                fig_angle_curve.update_yaxes(range=[0, max(100, float(max_flex or 90) + 10)])
                fig_angle_curve.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_angle_curve, use_container_width=True)
            else:
                st.info("Continuous angle timeline recorded across video playback.")

        # ── SECTION 4: 11 Parameters Display Table ──
        st.markdown("### 📋 The 11 Ergonomic & Drudgery Parameters")
        
        param_data = [
            {"#": "1", "Parameter": "Sitting Duration & Reps", "Extracted Value": f"{report.sitting_duration}s ({round(report.sitting_duration/max(1, report.total_tracked_time)*100, 1)}%) | Reps: {getattr(report, 'sitting_reps', 0)}"},
            {"#": "1b", "Parameter": "Squatting Duration & Reps", "Extracted Value": f"{getattr(report, 'squatting_duration', 0.0)}s ({round(getattr(report, 'squatting_duration', 0.0)/max(1, report.total_tracked_time)*100, 1)}%) | Reps: {getattr(report, 'squatting_reps', 0)}"},
            {"#": "2", "Parameter": "Standing Duration & Reps", "Extracted Value": f"{report.standing_duration}s ({round(report.standing_duration/max(1, report.total_tracked_time)*100, 1)}%) | Reps: {getattr(report, 'standing_reps', 0)}"},
            {"#": "3", "Parameter": "Bending Duration & Reps", "Extracted Value": f"{report.bending_duration}s ({round(report.bending_duration/max(1, report.total_tracked_time)*100, 1)}%) | Reps: {getattr(report, 'bending_reps', 0)} (Severe: {report.severe_bending_duration}s)"},
            {"#": "4", "Parameter": "Walking Duration", "Extracted Value": f"{report.walking_duration}s ({round(report.walking_duration/max(1, report.total_tracked_time)*100, 1)}%) | Reps: {getattr(report, 'walking_reps', 0)}"},
            {"#": "5", "Parameter": "Load Carried", "Extracted Value": f"{report.total_load_events} carrying events detected"},
            {"#": "6", "Parameter": "Repetitive Movement", "Extracted Value": f"{report.repetitive_movement.cycles_per_minute} cycles/min on {report.repetitive_movement.primary_joint} ({report.repetitive_movement.frequency_hz} Hz)" if report.repetitive_movement and report.repetitive_movement.is_repetitive else "None detected"},
            {"#": "7", "Parameter": "Number of Trips", "Extracted Value": f"{report.trip_count_result.trip_count if report.trip_count_result else 0} trips ({report.trip_count_result.total_distance_pixels if report.trip_count_result else 0} px travel)"},
            {"#": "8", "Parameter": "Tools/Equipment Used", "Extracted Value": ", ".join([t.tool_name for t in report.tools_used]) if report.tools_used else "None detected"},
            {"#": "9", "Parameter": "Posture & Angles (Trunk & Arms)", "Extracted Value": f"Trunk: {report.posture_summary.avg_trunk_flexion}° | Shoulder Elv: {getattr(report.posture_summary, 'avg_shoulder_angle', 'N/A')}° | Elbow: {getattr(report.posture_summary, 'avg_elbow_angle', 'N/A')}° | Knee: {getattr(report.posture_summary, 'avg_knee_angle', 'N/A')}°" if report.posture_summary else "N/A"},
            {"#": "10", "Parameter": "Continuous Work Duration", "Extracted Value": f"Longest: {report.longest_work_bout}s | Avg: {report.avg_work_bout}s ({len(report.work_bouts)} bouts)"},
            {"#": "11", "Parameter": "Rest Duration & Pauses", "Extracted Value": f"Total Rest: {report.total_rest_duration}s across {report.rest_count} rest periods"},
        ]
        
        st.dataframe(pd.DataFrame(param_data), use_container_width=True, hide_index=True)

        # ── Charts Section ──
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("#### Posture Distribution")
            if report.posture_summary and report.posture_summary.posture_distribution:
                dist = report.posture_summary.posture_distribution
                fig_pie = px.pie(
                    names=list(dist.keys()),
                    values=list(dist.values()),
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(fig_pie, use_container_width=True)

        with chart_col2:
            st.markdown("#### Activity Timeline Bouts")
            if report.activity_bouts:
                timeline_df = pd.DataFrame([
                    {
                        "Activity": bout.activity.value.title(),
                        "Start": bout.start_time,
                        "End": bout.end_time,
                        "Type": "Rest" if bout.is_rest else "Work"
                    }
                    for bout in report.activity_bouts
                ])
                fig_bar = px.timeline(
                    timeline_df,
                    x_start="Start",
                    x_end="End",
                    y="Activity",
                    color="Type",
                    color_discrete_map={"Work": "#3B82F6", "Rest": "#10B981"}
                )
                fig_bar.update_yaxes(autorange="reversed")
                st.plotly_chart(fig_bar, use_container_width=True)

        # ── 10-Minute Fatigue Progression Trend Line Chart ──
        fatigue_series = getattr(report, 'minute_fatigue_series', [])
        if fatigue_series:
            st.markdown("#### 📈 Cumulative Ergonomic Fatigue Curve (0 – 10+ Minutes)")
            fatigue_df = pd.DataFrame({
                "Minute": [f"Min {i+1}" for i in range(len(fatigue_series))],
                "Fatigue Level (%)": fatigue_series
            })
            fig_fatigue = px.line(
                fatigue_df,
                x="Minute",
                y="Fatigue Level (%)",
                markers=True,
                line_shape="spline",
                color_discrete_sequence=["#EF4444"]
            )
            fig_fatigue.update_yaxes(range=[0, 100])
            st.plotly_chart(fig_fatigue, use_container_width=True)

        # Export Buttons
        st.markdown("### 📥 Download Assessment Reports")
        ex_col1, ex_col2, ex_col3 = st.columns(3)
        with ex_col1:
            st.download_button(
                "📄 Download Publication PDF Report",
                data=result.pdf_report if result.pdf_report else b"",
                file_name=f"{Path(result.video_metadata.filename).stem}_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        with ex_col2:
            st.download_button(
                "📊 Download JSON Report",
                data=json.dumps(result.json_report, indent=2),
                file_name=f"{Path(result.video_metadata.filename).stem}_report.json",
                mime="application/json",
                use_container_width=True
            )
        with ex_col3:
            st.download_button(
                "📈 Download CSV Summary",
                data=result.csv_report,
                file_name=f"{Path(result.video_metadata.filename).stem}_report.csv",
                mime="text/csv",
                use_container_width=True
            )
