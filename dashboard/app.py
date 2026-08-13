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

        pipeline = AgriErgoPipeline(sample_fps=sample_fps)
        
        with st.spinner("Processing video..."):
            result = pipeline.process(video_path, progress_callback=update_progress)

        st.session_state["pipeline_result"] = result
        st.success(f"Analysis complete in {result.processing_time_seconds}s!")


# ──────────────────────────────────────────────
# Step 2: Display Results
# ──────────────────────────────────────────────
if "pipeline_result" in st.session_state:
    result = st.session_state["pipeline_result"]
    
    st.markdown("---")
    st.header("📊 Assessment Results")

    # Overview KPI Metrics
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Video Duration", f"{result.video_metadata.duration_seconds}s")
    with kpi2:
        st.metric("Workers Detected", result.workers_detected)
    with kpi3:
        st.metric("Processed Frames", result.frames_processed)
    with kpi4:
        st.metric("Processing Time", f"{result.processing_time_seconds}s")

    if not result.worker_reports:
        st.warning("No workers detected with sufficient frame history.")
    else:
        # Worker Selection Tabs
        worker_ids = [w.worker_id for w in result.worker_reports]
        selected_worker_id = st.selectbox("Select Worker to View Assessment", worker_ids, format_func=lambda x: f"Worker #{x}")

        report: WorkerReport = next(w for w in result.worker_reports if w.worker_id == selected_worker_id)

        st.subheader(f"Detailed Analysis for Worker #{selected_worker_id}")

        # Worker Overview Cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Tracked Time", f"{report.total_tracked_time}s")
        with c2:
            st.metric("Dominant Posture", report.posture_summary.dominant_posture.value.title() if report.posture_summary else "N/A")
        with c3:
            score_val = str(report.reba_score) if report.reba_score else "N/A"
            risk_lvl = report.reba_risk_level or "N/A"
            st.metric("REBA Risk Score", f"{score_val} ({risk_lvl})")
        with c4:
            st.metric("Trip Count", report.trip_count_result.trip_count if report.trip_count_result else 0)

        # ── 11 Parameters Display Table ──
        st.markdown("### 📋 The 11 Ergonomic & Drudgery Parameters")
        
        param_data = [
            {"#": 1, "Parameter": "Sitting Duration", "Extracted Value": f"{report.sitting_duration}s ({round(report.sitting_duration/max(1, report.total_tracked_time)*100, 1)}%)"},
            {"#": "1b", "Parameter": "Squatting Duration", "Extracted Value": f"{getattr(report, 'squatting_duration', 0.0)}s ({round(getattr(report, 'squatting_duration', 0.0)/max(1, report.total_tracked_time)*100, 1)}%)"},
            {"#": 2, "Parameter": "Standing Duration", "Extracted Value": f"{report.standing_duration}s ({round(report.standing_duration/max(1, report.total_tracked_time)*100, 1)}%)"},
            {"#": 3, "Parameter": "Bending Duration", "Extracted Value": f"{report.bending_duration}s ({round(report.bending_duration/max(1, report.total_tracked_time)*100, 1)}%)"},
            {"#": 4, "Parameter": "Walking Duration", "Extracted Value": f"{report.walking_duration}s ({round(report.walking_duration/max(1, report.total_tracked_time)*100, 1)}%)"},
            {"#": 5, "Parameter": "Load Carried", "Extracted Value": f"{report.total_load_events} carrying events detected"},
            {"#": 6, "Parameter": "Repetitive Movement", "Extracted Value": f"{report.repetitive_movement.cycles_per_minute} cycles/min on {report.repetitive_movement.primary_joint} ({report.repetitive_movement.frequency_hz} Hz)" if report.repetitive_movement and report.repetitive_movement.is_repetitive else "None detected"},
            {"#": 7, "Parameter": "Number of Trips", "Extracted Value": f"{report.trip_count_result.trip_count if report.trip_count_result else 0} trips ({report.trip_count_result.total_distance_pixels if report.trip_count_result else 0} px travel)"},
            {"#": 8, "Parameter": "Tools/Equipment Used", "Extracted Value": ", ".join([t.tool_name for t in report.tools_used]) if report.tools_used else "None detected"},
            {"#": 9, "Parameter": "Posture & Angles", "Extracted Value": f"Avg Trunk Flexion: {report.posture_summary.avg_trunk_flexion}° | Max: {report.posture_summary.max_trunk_flexion}°" if report.posture_summary else "N/A"},
            {"#": 10, "Parameter": "Continuous Work Duration", "Extracted Value": f"Longest: {report.longest_work_bout}s | Avg: {report.avg_work_bout}s ({len(report.work_bouts)} bouts)"},
            {"#": 11, "Parameter": "Rest Duration", "Extracted Value": f"Total Rest: {report.total_rest_duration}s across {report.rest_count} rest periods"},
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

        # Export Buttons
        st.markdown("### 📥 Download Reports")
        ex_col1, ex_col2 = st.columns(2)
        with ex_col1:
            st.download_button(
                "Download JSON Report",
                data=json.dumps(result.json_report, indent=2),
                file_name=f"{Path(result.video_metadata.filename).stem}_report.json",
                mime="application/json",
                use_container_width=True
            )
        with ex_col2:
            st.download_button(
                "Download CSV Summary",
                data=result.csv_report,
                file_name=f"{Path(result.video_metadata.filename).stem}_report.csv",
                mime="text/csv",
                use_container_width=True
            )
