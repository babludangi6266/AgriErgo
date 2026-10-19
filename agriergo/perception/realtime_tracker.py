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

        # FPS calculation
        self._last_frame_time: Optional[float] = None
        self._fps_history: deque = deque(maxlen=20)
        self.current_fps: float = 0.0
        self.frame_count: int = 0

    def reset(self):
        """Reset internal history and counters."""
        self._smoothed_angles.clear()
        self._posture_durations.clear()
        self._current_posture.clear()
        self._posture_streak_start.clear()
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

            # 8. Assemble Worker Telemetry
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

        return RealTimeFrameResult(
            annotated_frame=annotated,
            fps=self.current_fps,
            timestamp=now,
            workers=workers_telemetry,
            primary_telemetry=primary,
            alerts=global_alerts,
        )

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
