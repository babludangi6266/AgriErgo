"""
Real-Time Body Angle & Posture Tracker — High-Performance Live Biomechanical Stream Processor.

Processes live camera, webcam, or RTSP video streams frame-by-frame with low latency:
  1. Multi-person pose estimation & persistent ByteTrack tracking
  2. Sub-millisecond trigonometric joint angle computation (Trunk, Neck, Shoulder, Elbow, Hip, Knee)
  3. Exponential Moving Average (EMA) angle temporal smoothing
  4. Real-time posture classification (Bending, Squatting, Standing, Sitting, Walking)
  5. Instant REBA risk scoring & ISO 11226 ergonomic hazard alerting
  6. Biomechanical HUD video overlay with color-coded skeleton & live angle badges
"""

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import (
    POSE_MODEL,
    TRACKER_CONFIG,
    RESULTS_DIR,
    REALTIME_IMGSZ,
    REALTIME_CONFIDENCE,
    REALTIME_SMOOTHING_ALPHA,
    REALTIME_ALERT_TRUNK_SEVERE,
    REALTIME_ALERT_TRUNK_BENDING,
    REALTIME_ALERT_SHOULDER_OVERHEAD,
    REALTIME_ALERT_SHOULDER_ELEVATED,
    REALTIME_ALERT_NECK_STRAIN,
    REALTIME_ALERT_KNEE_SQUAT,
)
from agriergo.perception.pose_estimator import PoseEstimator, PersonKeypoints
from agriergo.perception.annotator import VideoAnnotator
from agriergo.interpretation.joint_angles import compute_joint_angles, JointAngles
from agriergo.interpretation.posture_classifier import PostureClassifier, PostureLabel
from agriergo.interpretation.biomechanics_3d import (
    Skeleton3D,
    SpineStress3D,
    compute_l5s1_spinal_compression,
    lift_2d_to_3d_skeleton,
)
from agriergo.analytics.ergonomic_scorer import ErgonomicScorer, REBAScore
from agriergo.analytics.pdf_generator import PDFReportGenerator


@dataclass
class WorkerLiveTelemetry:
    """Live telemetry payload for a single worker in the current frame."""
    worker_id: int
    posture: PostureLabel
    trunk_angle: float
    neck_angle: Optional[float] = None
    knee_angle: Optional[float] = None
    hip_angle: Optional[float] = None
    shoulder_angle: Optional[float] = None
    elbow_angle: Optional[float] = None

    # Raw (unsmoothed) angles
    raw_trunk_angle: Optional[float] = None
    raw_shoulder_angle: Optional[float] = None
    raw_knee_angle: Optional[float] = None

    # Ergonomic risk
    reba_score: int = 1
    reba_risk_level: str = "Negligible"

    # 3D Biomechanics & L5/S1 Spinal Stress
    l5s1_compression_n: float = 0.0
    l5s1_shear_n: float = 0.0
    l5s1_risk_level: str = "Safe"
    spine_stress: Optional[SpineStress3D] = None
    skeleton_3d: Optional[Skeleton3D] = None

    # Hazard flags
    is_severe_stoop: bool = False
    is_arm_overhead: bool = False
    is_arm_elevated: bool = False
    is_neck_strained: bool = False
    alerts: List[str] = field(default_factory=list)

    # Durations (seconds)
    time_in_current_posture: float = 0.0
    posture_durations: Dict[str, float] = field(default_factory=dict)
    bbox: Optional[np.ndarray] = None


@dataclass
class RealTimeFrameResult:
    """Comprehensive result of processing a single live frame."""
    annotated_frame: np.ndarray
    fps: float
    timestamp: float
    workers: List[WorkerLiveTelemetry]
    primary_telemetry: Optional[WorkerLiveTelemetry]
    alerts: List[str]
    audio_alert_triggered: bool = False
    audio_alert_message: Optional[str] = None



class RealTimePostureTracker:
    """
    Real-Time Body Angle and Posture Tracker for live video feeds, webcams, and RTSP streams.
    """

    def __init__(
        self,
        model_path: str = POSE_MODEL,
        confidence: float = REALTIME_CONFIDENCE,
        imgsz: int = REALTIME_IMGSZ,
        smoothing_alpha: float = REALTIME_SMOOTHING_ALPHA,
        enable_tracking: bool = True,
    ):
        self.confidence = confidence
        self.imgsz = imgsz
        self.smoothing_alpha = smoothing_alpha
        self.enable_tracking = enable_tracking

        # Perception & Analytics instances
        self.pose_estimator = PoseEstimator(model_path=model_path, confidence=confidence)
        self.posture_classifier = PostureClassifier(smoothing_window=3)
        self.ergonomic_scorer = ErgonomicScorer()

        # State tracking
        self._smoothed_angles: Dict[int, Dict[str, float]] = {}
        self._posture_durations: Dict[int, Dict[str, float]] = {}
        self._current_posture: Dict[int, PostureLabel] = {}
        self._posture_streak_start: Dict[int, float] = {}

        # Live Session Recording & Telemetry History
        self._session_start_time: float = time.time()
        self._session_records: List[Dict[str, Any]] = []

        # Audio Hazard Alert Debouncing
        self._last_audio_alert_time: float = 0.0
        self._audio_alert_cooldown: float = 4.0  # Seconds between audible warnings

        # Video Recording
        self._video_writer: Optional[cv2.VideoWriter] = None
        self._video_output_path: Optional[str] = None

        # PDF Generator
        self.pdf_generator = PDFReportGenerator()

        # FPS calculation
        self._last_frame_time: Optional[float] = None
        self._fps_history: deque = deque(maxlen=20)
        self.current_fps: float = 0.0
        self.frame_count: int = 0

    def reset(self):
        """Reset internal history, counters, and session records."""
        self._smoothed_angles.clear()
        self._posture_durations.clear()
        self._current_posture.clear()
        self._posture_streak_start.clear()
        self._session_start_time = time.time()
        self._session_records.clear()
        self._last_audio_alert_time = 0.0
        self.stop_video_recording()
        self._last_frame_time = None
        self._fps_history.clear()
        self.current_fps = 0.0
        self.frame_count = 0

    def process_frame(
        self,
        frame: np.ndarray,
        draw_overlay: bool = True,
        draw_hud: bool = True,
    ) -> RealTimeFrameResult:
        """
        Process a single frame from webcam, camera, or video stream.

        Args:
            frame: Input BGR image array (H, W, 3).
            draw_overlay: If True, draws skeleton, joints, and angle tags.
            draw_hud: If True, draws the glassmorphic telemetry HUD on the frame.

        Returns:
            RealTimeFrameResult containing the annotated frame, telemetry list, and alerts.
        """
        now = time.time()
        self.frame_count += 1

        # Calculate live delta time & FPS
        dt = 0.033  # Default ~30 FPS
        if self._last_frame_time is not None:
            dt = max(0.001, now - self._last_frame_time)
            fps_instant = 1.0 / dt
            self._fps_history.append(fps_instant)
            self.current_fps = float(np.mean(self._fps_history))
        self._last_frame_time = now

        # 1. Pose Estimation / Tracking
        if self.enable_tracking:
            persons = self.pose_estimator.track(frame, tracker=TRACKER_CONFIG, imgsz=self.imgsz)
        else:
            persons = self.pose_estimator.estimate(frame)

        workers_telemetry: List[WorkerLiveTelemetry] = []
        global_alerts: List[str] = []
        posture_labels_for_annotator = {}
        reba_scores_for_annotator = {}

        for idx, person in enumerate(persons):
            # Assign fallback worker_id if tracking not active or None
            wid = person.person_id if person.person_id is not None else (idx + 1)
            person.person_id = wid

            # 2. Compute Raw Joint Angles
            raw_angles = compute_joint_angles(person.keypoints, person.confidences, min_conf=self.confidence)

            # 3. Apply Exponential Moving Average (EMA) Angle Smoothing
            smoothed = self._smooth_angles(wid, raw_angles)

            # Construct smoothed JointAngles
            eff_angles = JointAngles(
                trunk_flexion=smoothed.get("trunk_flexion"),
                avg_hip_angle=smoothed.get("hip_angle"),
                avg_knee_angle=smoothed.get("knee_angle"),
                avg_elbow_angle=smoothed.get("elbow_angle"),
                neck_flexion=smoothed.get("neck_flexion"),
                avg_shoulder_angle=smoothed.get("shoulder_angle"),
                left_shoulder_angle=smoothed.get("shoulder_angle"),
                right_shoulder_angle=smoothed.get("shoulder_angle"),
            )

            # 4. Classify Posture
            posture_res = self.posture_classifier.classify(
                worker_id=wid,
                joint_angles=eff_angles,
                keypoints=person.keypoints,
                confidences=person.confidences,
            )
            posture = posture_res.label
            posture_labels_for_annotator[wid] = posture

            # 5. Track Posture Durations & Streaks
            self._update_durations(wid, posture, dt, now)
            streak_duration = now - self._posture_streak_start.get(wid, now)

            # 6. Ergonomic Risk Scoring (Instant REBA)
            reba_obj = self.ergonomic_scorer.score_frame(eff_angles)
            reba_scores_for_annotator[wid] = reba_obj.final_score

            # 7. Evaluate Ergonomic Hazard Alerts
            trunk_val = smoothed.get("trunk_flexion", 0.0) or 0.0
            sh_val = smoothed.get("shoulder_angle", 0.0) or 0.0
            knee_val = smoothed.get("knee_angle", 180.0) or 180.0
            neck_val = smoothed.get("neck_flexion", 0.0) or 0.0

            is_severe_stoop = trunk_val >= REALTIME_ALERT_TRUNK_SEVERE
            is_arm_overhead = sh_val >= REALTIME_ALERT_SHOULDER_OVERHEAD
            is_arm_elevated = sh_val >= REALTIME_ALERT_SHOULDER_ELEVATED
            is_neck_strained = neck_val >= REALTIME_ALERT_NECK_STRAIN

            worker_alerts = []
            if is_severe_stoop:
                worker_alerts.append(f"Worker #{wid}: SEVERE STOOPING ({round(trunk_val, 1)}°) — ISO 11226 Hazard")
            elif trunk_val >= REALTIME_ALERT_TRUNK_BENDING:
                worker_alerts.append(f"Worker #{wid}: Trunk Flexion Warning ({round(trunk_val, 1)}°)")

            if is_arm_overhead:
                worker_alerts.append(f"Worker #{wid}: OVERHEAD ARM HAZARD ({round(sh_val, 1)}°)")
            elif is_arm_elevated:
                worker_alerts.append(f"Worker #{wid}: Elevated Arm Posture ({round(sh_val, 1)}°)")

            if knee_val < REALTIME_ALERT_KNEE_SQUAT:
                worker_alerts.append(f"Worker #{wid}: Deep Squat Joint Stress ({round(knee_val, 1)}°)")

            if is_neck_strained:
                worker_alerts.append(f"Worker #{wid}: Neck Flexion Strain ({round(neck_val, 1)}°)")

            global_alerts.extend(worker_alerts)

            # 8. Compute 3D Biomechanical Skeleton & Spinal Load (L5/S1)
            spine_stress_obj = compute_l5s1_spinal_compression(
                trunk_flexion_deg=trunk_val,
                neck_flexion_deg=neck_val or 0.0,
            )
            skeleton_3d_obj = lift_2d_to_3d_skeleton(
                keypoints=person.keypoints,
                trunk_flexion_deg=trunk_val,
                neck_flexion_deg=neck_val or 0.0,
                knee_angle_deg=knee_val or 180.0,
            )

            # Record telemetry to session history buffer
            record = {
                "timestamp": now,
                "relative_time_seconds": round(now - self._session_start_time, 2),
                "frame_idx": self.frame_count,
                "worker_id": wid,
                "posture": posture.value,
                "trunk_angle_deg": round(trunk_val, 2),
                "neck_angle_deg": round(neck_val, 2) if neck_val is not None else None,
                "shoulder_angle_deg": round(sh_val, 2) if sh_val is not None else None,
                "elbow_angle_deg": round(smoothed.get("elbow_angle", 0.0), 2) if smoothed.get("elbow_angle") is not None else None,
                "knee_angle_deg": round(knee_val, 2) if knee_val is not None else None,
                "hip_angle_deg": round(smoothed.get("hip_angle", 0.0), 2) if smoothed.get("hip_angle") is not None else None,
                "reba_score": reba_obj.final_score,
                "reba_risk_level": reba_obj.risk_level,
                "l5s1_compression_n": spine_stress_obj.l5s1_compression_n,
                "l5s1_shear_n": spine_stress_obj.l5s1_shear_n,
                "l5s1_risk_level": spine_stress_obj.risk_level,
                "is_severe_stoop": is_severe_stoop,
                "is_arm_overhead": is_arm_overhead,
                "is_arm_elevated": is_arm_elevated,
            }
            self._session_records.append(record)

            telem = WorkerLiveTelemetry(
                worker_id=wid,
                posture=posture,
                trunk_angle=trunk_val,
                neck_angle=smoothed.get("neck_flexion"),
                knee_angle=smoothed.get("knee_angle"),
                hip_angle=smoothed.get("hip_angle"),
                shoulder_angle=smoothed.get("shoulder_angle"),
                elbow_angle=smoothed.get("elbow_angle"),
                raw_trunk_angle=raw_angles.trunk_flexion,
                raw_shoulder_angle=raw_angles.avg_shoulder_angle,
                raw_knee_angle=raw_angles.avg_knee_angle,
                reba_score=reba_obj.final_score,
                reba_risk_level=reba_obj.risk_level,
                l5s1_compression_n=spine_stress_obj.l5s1_compression_n,
                l5s1_shear_n=spine_stress_obj.l5s1_shear_n,
                l5s1_risk_level=spine_stress_obj.risk_level,
                spine_stress=spine_stress_obj,
                skeleton_3d=skeleton_3d_obj,
                is_severe_stoop=is_severe_stoop,
                is_arm_overhead=is_arm_overhead,
                is_arm_elevated=is_arm_elevated,
                is_neck_strained=is_neck_strained,
                alerts=worker_alerts,
                time_in_current_posture=streak_duration,
                posture_durations=dict(self._posture_durations.get(wid, {})),
                bbox=person.bbox,
            )
            workers_telemetry.append(telem)

        # 9. Frame Visual Annotation
        annotated = frame.copy()
        if draw_overlay and len(persons) > 0:
            annotated = VideoAnnotator.annotate_frame(
                annotated,
                persons,
                posture_labels=posture_labels_for_annotator,
                reba_scores=reba_scores_for_annotator,
            )

        # 10. Draw Biomechanical Telemetry HUD Card
        primary = workers_telemetry[0] if workers_telemetry else None
        if draw_hud and primary is not None:
            hud_telem = {
                "posture": primary.posture,
                "trunk_angle": primary.trunk_angle,
                "shoulder_angle": primary.shoulder_angle,
                "knee_angle": primary.knee_angle,
                "hip_angle": primary.hip_angle,
                "neck_angle": primary.neck_angle,
                "elbow_angle": primary.elbow_angle,
                "reba_score": primary.reba_score,
                "risk_level": primary.reba_risk_level,
            }
            annotated = VideoAnnotator.draw_realtime_hud(
                annotated,
                hud_telem,
                fps=self.current_fps,
                alerts=primary.alerts,
            )

        # 11. Evaluate Audible Hazard Alert Trigger
        audio_triggered = False
        audio_msg = None
        if primary is not None and (primary.is_severe_stoop or primary.is_arm_overhead):
            if now - self._last_audio_alert_time >= self._audio_alert_cooldown:
                self._last_audio_alert_time = now
                audio_triggered = True
                if primary.is_severe_stoop:
                    audio_msg = f"Warning: Severe stooping posture ({round(primary.trunk_angle, 1)} degrees)! Straighten your trunk."
                else:
                    audio_msg = f"Warning: Overhead arm strain ({round(primary.shoulder_angle, 1)} degrees)! Lower your arms."

        # 12. Write to MP4 video writer if recording active
        if self._video_writer is not None:
            self._video_writer.write(annotated)

        return RealTimeFrameResult(
            annotated_frame=annotated,
            fps=self.current_fps,
            timestamp=now,
            workers=workers_telemetry,
            primary_telemetry=primary,
            alerts=global_alerts,
            audio_alert_triggered=audio_triggered,
            audio_alert_message=audio_msg,
        )

    def start_video_recording(
        self,
        output_path: Optional[str] = None,
        fps: float = 20.0,
        frame_size: Tuple[int, int] = (640, 480),
    ) -> str:
        """Initialize OpenCV VideoWriter to record the live annotated session."""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        if output_path is None:
            timestamp = int(time.time())
            output_path = str(RESULTS_DIR / f"live_session_{timestamp}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        self._video_output_path = output_path
        return output_path

    def stop_video_recording(self) -> Optional[str]:
        """Stop and close the active video writer."""
        if self._video_writer is not None:
            try:
                self._video_writer.release()
            except Exception:
                pass
            self._video_writer = None
            path = self._video_output_path
            self._video_output_path = None
            return path
        return None

    def get_session_dataframe(self) -> pd.DataFrame:
        """Return all recorded session telemetry records as a pandas DataFrame."""
        if not self._session_records:
            return pd.DataFrame()
        return pd.DataFrame(self._session_records)

    def export_session_csv(self, output_path: Optional[str] = None) -> str:
        """Export session telemetry records to CSV."""
        df = self.get_session_dataframe()
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        if output_path is None:
            timestamp = int(time.time())
            output_path = str(RESULTS_DIR / f"live_session_{timestamp}_telemetry.csv")

        df.to_csv(output_path, index=False)
        return output_path

    def get_session_summary(self, source_desc: str = "Live Stream") -> Dict[str, Any]:
        """
        Compute aggregate summary statistics from the live tracking session.
        """
        df = self.get_session_dataframe()
        total_time = max(1.0, time.time() - self._session_start_time)
        hrs = int(total_time // 3600)
        mins = int((total_time % 3600) // 60)
        secs = int(total_time % 60)
        time_formatted = f"{hrs:02d}:{mins:02d}:{secs:02d}"

        if df.empty:
            return {
                "session_duration_seconds": total_time,
                "session_duration_formatted": time_formatted,
                "total_frames": self.frame_count,
                "avg_fps": self.current_fps,
                "source_desc": source_desc,
                "posture_distribution": {},
                "avg_trunk_angle": 0.0,
                "max_trunk_angle": 0.0,
                "severe_stoop_seconds": 0.0,
                "severe_stoop_pct": 0.0,
                "avg_shoulder_angle": 0.0,
                "max_shoulder_angle": 0.0,
                "overhead_arm_seconds": 0.0,
                "overhead_arm_pct": 0.0,
                "avg_knee_angle": 170.0,
                "min_knee_angle": 170.0,
                "squatting_seconds": 0.0,
                "avg_neck_angle": 0.0,
                "max_neck_angle": 0.0,
                "neck_strain_seconds": 0.0,
                "peak_reba_score": 1,
                "dominant_risk_level": "Low",
                "iso_11226_violation": False,
                "recommendations": [
                    "Maintain neutral upright trunk posture (<20° flexion).",
                    "Avoid prolonged overhead reaching (>90° shoulder elevation).",
                    "Take regular micro-breaks during manual field cultivation.",
                ],
            }

        # Calculate durations from records
        stoop_count = int(df["is_severe_stoop"].sum())
        overhead_count = int(df["is_arm_overhead"].sum())
        total_records = max(1, len(df))

        severe_stoop_pct = (stoop_count / total_records) * 100.0
        overhead_arm_pct = (overhead_count / total_records) * 100.0
        severe_stoop_seconds = (stoop_count / total_records) * total_time
        overhead_arm_seconds = (overhead_count / total_records) * total_time

        # Posture distribution
        p_counts = df["posture"].value_counts()
        posture_dist = {}
        for p_name, count in p_counts.items():
            pct = (count / total_records) * 100.0
            dur_s = (count / total_records) * total_time
            posture_dist[p_name] = {
                "duration_seconds": round(dur_s, 1),
                "percentage": round(pct, 1),
            }

        squat_dur = posture_dist.get("squatting", {}).get("duration_seconds", 0.0)

        # Joint angle averages & peaks
        avg_trunk = float(df["trunk_angle_deg"].mean())
        max_trunk = float(df["trunk_angle_deg"].max())
        avg_sh = float(df["shoulder_angle_deg"].dropna().mean()) if "shoulder_angle_deg" in df else 0.0
        max_sh = float(df["shoulder_angle_deg"].dropna().max()) if "shoulder_angle_deg" in df else 0.0
        avg_kn = float(df["knee_angle_deg"].dropna().mean()) if "knee_angle_deg" in df else 170.0
        min_kn = float(df["knee_angle_deg"].dropna().min()) if "knee_angle_deg" in df else 170.0
        avg_nk = float(df["neck_angle_deg"].dropna().mean()) if "neck_angle_deg" in df else 0.0
        max_nk = float(df["neck_angle_deg"].dropna().max()) if "neck_angle_deg" in df else 0.0

        # L5/S1 Spinal Compression metrics
        avg_l5s1 = float(df["l5s1_compression_n"].mean()) if "l5s1_compression_n" in df else 0.0
        max_l5s1 = float(df["l5s1_compression_n"].max()) if "l5s1_compression_n" in df else 0.0
        l5s1_exceeded_count = int((df["l5s1_compression_n"] >= 3400.0).sum()) if "l5s1_compression_n" in df else 0
        l5s1_exceeded_pct = (l5s1_exceeded_count / total_records) * 100.0

        peak_reba = int(df["reba_score"].max())
        risk_counts = df["reba_risk_level"].value_counts()
        dominant_risk = str(risk_counts.index[0]) if not risk_counts.empty else "Low"

        iso_violation = severe_stoop_seconds >= 60.0 or severe_stoop_pct >= 25.0

        # Build prioritized actionable recommendations
        recs = []
        if max_l5s1 >= 3400.0 or l5s1_exceeded_pct > 15.0:
            recs.append("L5/S1 spinal load exceeded NIOSH 3400 N threshold: Integrate lumbar-assist exosuits or rolling plant stools to protect lumbar disc endplates.")
        if severe_stoop_pct > 20.0 or max_trunk > 60.0:
            recs.append("Severe stooping detected: Utilize long-handled cultivation tools or rolling field seats to reduce lumbar spine stress.")
        if overhead_arm_pct > 15.0 or max_sh > 90.0:
            recs.append("Overhead arm reach detected: Lower harvesting heights or rotate arm tasks to prevent rotator cuff impingement.")
        if squat_dur > 60.0:
            recs.append("Prolonged squatting: Provide padded knee support or mobile farm stools to relieve patellofemoral compressive force.")
        recs.append("Schedule 2-minute micro-breaks every 20 minutes to restore joint mobility.")

        return {
            "session_duration_seconds": total_time,
            "session_duration_formatted": time_formatted,
            "total_frames": self.frame_count,
            "avg_fps": float(np.mean(self._fps_history)) if self._fps_history else self.current_fps,
            "source_desc": source_desc,
            "posture_distribution": posture_dist,
            "avg_trunk_angle": avg_trunk,
            "max_trunk_angle": max_trunk,
            "avg_l5s1_compression_n": avg_l5s1,
            "max_l5s1_compression_n": max_l5s1,
            "l5s1_action_limit_pct": l5s1_exceeded_pct,
            "severe_stoop_seconds": severe_stoop_seconds,
            "severe_stoop_pct": severe_stoop_pct,
            "avg_shoulder_angle": avg_sh,
            "max_shoulder_angle": max_sh,
            "overhead_arm_seconds": overhead_arm_seconds,
            "overhead_arm_pct": overhead_arm_pct,
            "avg_knee_angle": avg_kn,
            "min_knee_angle": min_kn,
            "squatting_seconds": squat_dur,
            "avg_neck_angle": avg_nk,
            "max_neck_angle": max_nk,
            "neck_strain_seconds": float((df["neck_angle_deg"].dropna() > 25).sum() / total_records * total_time) if "neck_angle_deg" in df else 0.0,
            "peak_reba_score": peak_reba,
            "dominant_risk_level": dominant_risk,
            "iso_11226_violation": iso_violation,
            "recommendations": recs,
        }


    def export_session_pdf(self, output_path: Optional[str] = None, source_desc: str = "Live Stream") -> bytes:
        """Export session summary report to a publication-ready PDF."""
        summary = self.get_session_summary(source_desc=source_desc)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        if output_path is None:
            timestamp = int(time.time())
            output_path = str(RESULTS_DIR / f"live_session_{timestamp}_report.pdf")

        return self.pdf_generator.generate_live_session_pdf(summary, output_path=output_path)

    def _smooth_angles(self, wid: int, raw: JointAngles) -> Dict[str, float]:
        """Apply Exponential Moving Average smoothing on joint angles."""
        if wid not in self._smoothed_angles:
            self._smoothed_angles[wid] = {}

        prev = self._smoothed_angles[wid]
        alpha = self.smoothing_alpha
        smoothed = {}

        mapping = {
            "trunk_flexion": raw.trunk_flexion,
            "neck_flexion": raw.neck_flexion,
            "knee_angle": raw.avg_knee_angle,
            "hip_angle": raw.avg_hip_angle,
            "shoulder_angle": raw.avg_shoulder_angle or raw.max_shoulder_angle,
            "elbow_angle": raw.avg_elbow_angle,
        }

        for k, v in mapping.items():
            if v is not None:
                if k in prev and prev[k] is not None:
                    smoothed[k] = alpha * float(v) + (1.0 - alpha) * prev[k]
                else:
                    smoothed[k] = float(v)
            else:
                smoothed[k] = prev.get(k)

        self._smoothed_angles[wid] = smoothed
        return smoothed

    def _update_durations(self, wid: int, posture: PostureLabel, dt: float, now: float):
        """Update cumulative and streak timers for worker posture."""
        if wid not in self._posture_durations:
            self._posture_durations[wid] = {p.value: 0.0 for p in PostureLabel}

        p_str = posture.value if hasattr(posture, 'value') else str(posture)
        self._posture_durations[wid][p_str] = self._posture_durations[wid].get(p_str, 0.0) + dt

        prev_posture = self._current_posture.get(wid)
        if prev_posture != posture:
            self._current_posture[wid] = posture
            self._posture_streak_start[wid] = now
