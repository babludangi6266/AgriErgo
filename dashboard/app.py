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
import cv2
from collections import deque

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (
    STREAMLIT_PAGE_TITLE,
    STREAMLIT_PAGE_ICON,
    SUPPORTED_FORMATS,
    MAX_UPLOAD_SIZE_MB,
    REALTIME_IMGSZ,
    REALTIME_CONFIDENCE,
)
from agriergo.pipeline import AgriErgoPipeline
from agriergo.analytics.parameter_aggregator import WorkerReport
from agriergo.perception.realtime_tracker import RealTimePostureTracker


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
# ──────────────────────────────────────────────
# Sidebar - Navigation & Workflow Selector
# ──────────────────────────────────────────────
st.sidebar.title("🎯 Workflow Mode")
app_mode = st.sidebar.radio(
    "Choose Assessment Mode:",
    [
        "📹 Real-Time Live Tracking (Camera / Stream)",
        "📁 Video File Batch Analysis (Full Pipeline)",
    ],
    index=0,
    help="Switch between live body angle tracking and full video file processing."
)
st.sidebar.markdown("---")

@st.cache_resource
def get_pipeline():
    """Cache AgriErgoPipeline model instances in RAM for 0s instant startup."""
    return AgriErgoPipeline()


# ==============================================================================
# WORKFLOW 1: REAL-TIME LIVE POSTURE & BODY ANGLE TRACKING
# ==============================================================================
if app_mode == "📹 Real-Time Live Tracking (Camera / Stream)":
    st.subheader("📹 Real-Time Body Angle & Posture Tracking")
    st.caption(
        "Live high-framerate stream tracking of Trunk Flexion, Neck Angle, Shoulder Elevation, "
        "Elbow & Knee Kinematics with real-time HUD and ISO 11226 hazard detection."
    )

    with st.sidebar:
        st.header("⚙️ Stream Configuration")
        stream_type = st.radio(
            "Input Source",
            ["🎥 Webcam / USB Camera", "🎞️ Stream Field Video (Simulated Live)", "🌐 RTSP / Network Stream"],
            index=0,
        )

        stream_src = 0
        if stream_type == "🎥 Webcam / USB Camera":
            cam_index = st.number_input(
                "Camera Device Index",
                min_value=0,
                max_value=5,
                value=0,
                help="0 is usually built-in webcam. 1, 2 for external USB cameras."
            )
            stream_src = int(cam_index)
        elif stream_type == "🎞️ Stream Field Video (Simulated Live)":
            existing_videos = list(Path("data/uploads").glob("*.mp4"))
            if existing_videos:
                chosen = st.selectbox("Select Existing Field Video", [str(v) for v in existing_videos])
                stream_src = chosen
            else:
                up_demo = st.file_uploader("Upload video to stream", type=["mp4", "avi", "mov"])
                if up_demo:
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    tfile.write(up_demo.getvalue())
                    tfile.close()
                    stream_src = tfile.name
                else:
                    stream_src = None
        else:
            stream_src = st.text_input("RTSP Stream URL", placeholder="rtsp://192.168.1.50:554/live")

        rt_conf = st.slider("Keypoint Confidence", min_value=0.20, max_value=0.70, value=float(REALTIME_CONFIDENCE), step=0.05)
        rt_imgsz = st.selectbox("Inference Resolution", [256, 320, 416], index=1, help="Lower resolution runs at higher FPS")
        show_hud = st.checkbox("Show Glassmorphic HUD Overlay", value=True)
        show_chart = st.checkbox("Show Real-Time Angle Strip Chart", value=True)
        enable_audio = st.checkbox("🔊 Enable Audible Hazard Alerts (Voice & Beep)", value=True, help="Plays audible warning tone & voice when severe stooping or overhead work is detected")
        record_video = st.checkbox("🔴 Record Live Video Session to MP4", value=False, help="Saves live annotated feed as an MP4 video file")

    # Control Action Buttons
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2.5, 1, 1])
    with ctrl_col1:
        st.info("💡 **Ergonomic Standards**: Severe Stoop (>60° trunk flexion), Awkward Arm (>45° shoulder), Overhead Strain (>90°), Squat Stress (<90° knee).")
    with ctrl_col2:
        start_btn = st.button("▶️ Start Live Tracking", type="primary", use_container_width=True)
    with ctrl_col3:
        stop_btn = st.button("⏹️ Stop Tracking", use_container_width=True)

    if start_btn:
        st.session_state["live_tracking_active"] = True
    if stop_btn:
        st.session_state["live_tracking_active"] = False

    if st.session_state.get("live_tracking_active", False):
        if stream_src is None:
            st.error("Please select or upload a valid video stream source.")
        else:
            tracker = RealTimePostureTracker(
                confidence=rt_conf,
                imgsz=rt_imgsz,
                enable_tracking=True,
            )

            # Open video capture stream
            if isinstance(stream_src, int):
                cap = cv2.VideoCapture(stream_src, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(stream_src)
            else:
                cap = cv2.VideoCapture(str(stream_src))

            if not cap.isOpened():
                st.error(f"❌ Failed to open video source: `{stream_src}`. Please verify camera connection or permissions.")
                st.session_state["live_tracking_active"] = False
            else:
                kpi_placeholder = st.empty()
                audio_placeholder = st.empty()
                left_col, right_col = st.columns([1.6, 1.0])
                with left_col:
                    video_placeholder = st.empty()
                    alert_placeholder = st.empty()
                with right_col:
                    chart_placeholder = st.empty()
                    posture_breakdown_placeholder = st.empty()

                hist_frames = deque(maxlen=40)
                hist_trunk = deque(maxlen=40)
                hist_shoulder = deque(maxlen=40)
                hist_knee = deque(maxlen=40)
                hist_neck = deque(maxlen=40)

                frame_idx = 0
                is_file = not isinstance(stream_src, int) and not str(stream_src).startswith("rtsp")
                writer_started = False

                while cap.isOpened() and st.session_state.get("live_tracking_active", False):
                    ret, frame = cap.read()
                    if not ret:
                        if is_file:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        else:
                            st.warning("⚠️ Live stream ended or disconnected.")
                            break

                    frame_idx += 1

                    # Initialize video writer if recording requested
                    if record_video and not writer_started:
                        h_f, w_f = frame.shape[:2]
                        tracker.start_video_recording(fps=20.0, frame_size=(w_f, h_f))
                        writer_started = True

                    result = tracker.process_frame(frame, draw_overlay=True, draw_hud=show_hud)
                    annotated_rgb = cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)

                    primary = result.primary_telemetry
                    if primary is not None:
                        # 1. Update Real-Time Metric Cards
                        with kpi_placeholder.container():
                            k1, k2, k3, k4, k5, k6 = st.columns(6)

                            p_name = primary.posture.value.title()
                            if primary.is_severe_stoop:
                                p_badge = f"🔴 {p_name}"
                            elif primary.posture.value == "squatting":
                                p_badge = f"🟡 {p_name}"
                            else:
                                p_badge = f"🟢 {p_name}"
                            k1.metric("Live Posture", p_badge)

                            t_deg = round(primary.trunk_angle, 1)
                            t_delta = "Severe Stoop!" if primary.is_severe_stoop else ("Bending" if t_deg > 30 else "Upright")
                            k2.metric("Trunk Flexion", f"{t_deg}°", delta=t_delta, delta_color="inverse" if primary.is_severe_stoop else "normal")

                            sh_deg = round(primary.shoulder_angle, 1) if primary.shoulder_angle is not None else 0.0
                            sh_delta = "Overhead Hazard" if primary.is_arm_overhead else ("Elevated" if primary.is_arm_elevated else "Normal")
                            k3.metric("Shoulder Angle", f"{sh_deg}°", delta=sh_delta, delta_color="inverse" if primary.is_arm_overhead else "normal")

                            kn_deg = round(primary.knee_angle, 1) if primary.knee_angle is not None else 180.0
                            kn_delta = "Squatting" if kn_deg < 100 else "Standing"
                            k4.metric("Knee Angle", f"{kn_deg}°", delta=kn_delta)

                            nk_deg = round(primary.neck_angle, 1) if primary.neck_angle is not None else 0.0
                            k5.metric("Neck Flexion", f"{nk_deg}°")

                            k6.metric("Instant REBA", f"Score {primary.reba_score}", delta=primary.reba_risk_level, delta_color="inverse" if primary.reba_score > 7 else "normal")

                        # 2. Update Active Alerts & Audible Warning
                        if primary.alerts:
                            with alert_placeholder.container():
                                for alt in primary.alerts[:2]:
                                    st.error(f"⚠️ **ERGONOMIC HAZARD DETECTED**: {alt}")
                        else:
                            alert_placeholder.empty()

                        # Trigger Audible Warning if hazard detected and enabled
                        if result.audio_alert_triggered and enable_audio:
                            audio_js = f"""
                            <script>
                            try {{
                                const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                                const osc = audioCtx.createOscillator();
                                const gain = audioCtx.createGain();
                                osc.type = "sawtooth";
                                osc.frequency.setValueAtTime(880, audioCtx.currentTime);
                                gain.gain.setValueAtTime(0.25, audioCtx.currentTime);
                                gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.35);
                                osc.connect(gain);
                                gain.connect(audioCtx.destination);
                                osc.start();
                                osc.stop(audioCtx.currentTime + 0.35);

                                if ('speechSynthesis' in window) {{
                                    const msg = new SpeechSynthesisUtterance("{result.audio_alert_message or 'Warning: Ergonomic Hazard Detected'}");
                                    msg.rate = 1.1;
                                    window.speechSynthesis.speak(msg);
                                }}
                            }} catch(e) {{}}
                            </script>
                            """
                            with audio_placeholder.container():
                                st.components.v1.html(audio_js, height=0)

                        # 3. Update Rolling Strip Chart
                        if show_chart:
                            hist_frames.append(frame_idx)
                            hist_trunk.append(primary.trunk_angle)
                            hist_shoulder.append(primary.shoulder_angle or 0.0)
                            hist_knee.append(primary.knee_angle or 180.0)
                            hist_neck.append(primary.neck_angle or 0.0)

                            if len(hist_frames) > 2:
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=list(hist_frames), y=list(hist_trunk), mode='lines', name='Trunk Flexion', line=dict(color='#EF4444', width=2.5)))
                                fig.add_trace(go.Scatter(x=list(hist_frames), y=list(hist_shoulder), mode='lines', name='Shoulder Elev.', line=dict(color='#F59E0B', width=2)))
                                fig.add_trace(go.Scatter(x=list(hist_frames), y=list(hist_knee), mode='lines', name='Knee Angle', line=dict(color='#10B981', width=1.5)))

                                # Hazard threshold reference lines
                                fig.add_hline(y=60, line_dash="dash", line_color="#EF4444", annotation_text="Severe Stoop (60°)")
                                fig.add_hline(y=45, line_dash="dot", line_color="#F59E0B", annotation_text="Arm Strain (45°)")
                                fig.add_hline(y=20, line_dash="dot", line_color="#10B981", annotation_text="Upright (20°)")

                                fig.update_layout(
                                    title="Live Biomechanical Angle Stream (°)",
                                    height=250,
                                    margin=dict(l=20, r=20, t=35, b=20),
                                    legend=dict(orientation="h", y=1.15),
                                    yaxis=dict(range=[0, 190]),
                                )
                                chart_placeholder.plotly_chart(fig, use_container_width=True)

                        # 4. Posture Time Breakdown
                        with posture_breakdown_placeholder.container():
                            st.markdown("##### ⏱️ Cumulative Posture Time")
                            durs = primary.posture_durations
                            p_df = pd.DataFrame([
                                {"Posture": k.title(), "Seconds": round(v, 1)}
                                for k, v in durs.items() if v > 0.5
                            ])
                            if not p_df.empty:
                                fig_bar = px.bar(p_df, x="Seconds", y="Posture", orientation='h', color="Posture", height=160)
                                fig_bar.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
                                st.plotly_chart(fig_bar, use_container_width=True)

                    time.sleep(0.015)

                cap.release()

                # Save completed session data to session_state
                rec_video_path = tracker.stop_video_recording() if record_video else None
                summary_data = tracker.get_session_summary(source_desc=stream_type)
                df_session = tracker.get_session_dataframe()
                pdf_bytes = tracker.export_session_pdf(source_desc=stream_type)

                st.session_state["completed_live_summary"] = summary_data
                st.session_state["completed_live_df"] = df_session
                st.session_state["completed_live_pdf"] = pdf_bytes
                st.session_state["completed_live_video"] = rec_video_path

    # Display Live Session Summary & Download Export Center
    if "completed_live_summary" in st.session_state:
        st.markdown("---")
        st.header("📋 Live Session Completed — Export Center")
        s = st.session_state["completed_live_summary"]

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Tracked Duration", s["session_duration_formatted"])
        m2.metric("Frames Analyzed", s["total_frames"])
        m3.metric("Avg FPS", f"{round(s['avg_fps'], 1)}")
        m4.metric("Mean Trunk Flexion", f"{round(s['avg_trunk_angle'], 1)}°")
        m5.metric("Peak REBA Score", f"Score {s['peak_reba_score']}")
        m6.metric("ISO 11226 Status", "VIOLATION" if s["iso_11226_violation"] else "Compliant", delta_color="inverse" if s["iso_11226_violation"] else "normal")

        # Download Buttons Row
        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            df_export = st.session_state.get("completed_live_df", pd.DataFrame())
            if not df_export.empty:
                csv_data = df_export.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Session Data (CSV)",
                    data=csv_data,
                    file_name=f"agriergo_live_session_{int(time.time())}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        with dl2:
            pdf_data = st.session_state.get("completed_live_pdf", None)
            if pdf_data is not None:
                st.download_button(
                    "📄 Download Assessment Report (PDF)",
                    data=pdf_data,
                    file_name=f"agriergo_live_assessment_{int(time.time())}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
        with dl3:
            vid_path = st.session_state.get("completed_live_video", None)
            if vid_path and Path(vid_path).exists():
                with open(vid_path, "rb") as f_vid:
                    vid_bytes = f_vid.read()
                st.download_button(
                    "🎬 Download Recorded Video (MP4)",
                    data=vid_bytes,
                    file_name=Path(vid_path).name,
                    mime="video/mp4",
                    use_container_width=True,
                )

        # Telemetry Data Preview Expander
        df_preview = st.session_state.get("completed_live_df", pd.DataFrame())
        if not df_preview.empty:
            with st.expander("🔍 View Live Telemetry Time-Series Preview (First 20 Frames)"):
                st.dataframe(df_preview.head(20), use_container_width=True)


# ==============================================================================
# WORKFLOW 2: BATCH VIDEO FILE ANALYSIS (FULL PIPELINE)
# ==============================================================================
else:
    # Sidebar Configuration for Batch Video Mode
    with st.sidebar:
        st.header("⚙️ Batch Pipeline Settings")
        speed_mode = st.radio(
            "⚡ Processing Speed Mode",
            ["⚡ Lightning Fast (<5s)", "⚖️ Balanced Fast (Default)", "🔬 High Precision Research"],
            index=1,
            help="Lightning Fast processes 10-30 min videos in 3-5s. Balanced Fast processes in 8-12s."
        )

        sample_fps = 5.0

        # AI / Gemini Configuration
        st.markdown("---")
        st.markdown("### 🧠 AI Analysis (Gemini)")
        enable_ai = st.toggle(
            "🤖 Enable AI Analysis",
            value=False,
            help="Use Google Gemini AI to analyze video frames for intelligent task classification, risk narratives, and recommendations."
        )
        gemini_api_key = ""
        if enable_ai:
            gemini_api_key = st.text_input(
                "🔑 Gemini API Key",
                type="password",
                placeholder="Enter your Google Gemini API key",
                help="Get your free API key from https://aistudio.google.com/apikey"
            )
            if not gemini_api_key:
                st.warning("⚠️ Enter your Gemini API key to enable AI analysis.")
            else:
                st.success("✅ API key configured")

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

    uploaded_file = st.file_uploader(
        "Upload Field-Work Video",
        type=[fmt.replace('.', '') for fmt in SUPPORTED_FORMATS],
        help="Upload an MP4, AVI, or MOV video clip of agricultural field work."
    )

    if uploaded_file is not None:
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
                        enable_ai=enable_ai and bool(gemini_api_key),
                        gemini_api_key=gemini_api_key if enable_ai else None,
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

        # ── AI Ergonomic Insights Panel ──
        if getattr(report, 'ai_analysis_success', False):
            with st.expander("🧠 AI Ergonomic Insights (Gemini)", expanded=True):
                ai_col1, ai_col2, ai_col3, ai_col4 = st.columns(4)
                with ai_col1:
                    ai_conf = round((report.ai_task_confidence or 0) * 100)
                    st.metric("🎯 AI Task", report.ai_task_classification or "N/A", help=f"Confidence: {ai_conf}%")
                with ai_col2:
                    st.metric("🌾 Crop Detected", report.ai_crop_detected or "N/A")
                with ai_col3:
                    severity = report.ai_severity_rating or "N/A"
                    severity_colors = {"Low": "🟢", "Moderate": "🟡", "High": "🟠", "Critical": "🔴"}
                    severity_icon = severity_colors.get(severity, "⚪")
                    st.metric(f"{severity_icon} AI Severity", severity)
                with ai_col4:
                    st.metric("🏞️ Terrain", report.ai_terrain_type or "N/A")

                # Risk Narrative
                if report.ai_risk_narrative:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #1E1B4B, #312E81); padding: 16px; border-radius: 10px; border-left: 4px solid #818CF8; margin: 8px 0;">
                        <p style="color: #C7D2FE; font-size: 0.85rem; font-weight: 600; margin-bottom: 4px;">🧠 AI RISK ASSESSMENT</p>
                        <p style="color: #E0E7FF; font-size: 0.95rem; line-height: 1.5;">{report.ai_risk_narrative}</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Working Conditions
                if report.ai_working_conditions:
                    st.caption(f"🌤️ **Field Conditions:** {report.ai_working_conditions}")

                # AI Tools vs Detection Tools comparison
                ai_tools = report.ai_tools_identified or []
                if ai_tools:
                    st.caption(f"🔧 **AI-Identified Tools:** {', '.join(ai_tools)}")

                # Recommendations
                if report.ai_recommendations:
                    st.markdown("**💡 AI Recommendations:**")
                    for i, rec in enumerate(report.ai_recommendations, 1):
                        st.markdown(f"  {i}. {rec}")

                st.caption(f"Model: `{report.ai_model_used}` | AI Confidence: {ai_conf}%")

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
