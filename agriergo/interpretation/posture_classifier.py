"""
Posture Classifier — Rule-based classification of body posture.

Converts computed joint angles and displacement data into posture labels:
SITTING, STANDING, BENDING, WALKING, UNKNOWN.

Includes temporal smoothing via majority-vote to reduce label flickering.
"""

import numpy as np
from enum import Enum
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Dict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    TRUNK_FLEXION_BENDING,
    TRUNK_FLEXION_UPRIGHT,
    HIP_ANGLE_SITTING,
    HIP_ANGLE_STANDING,
    WALKING_DISPLACEMENT_THRESHOLD,
    WALKING_MIN_CONSECUTIVE,
    POSTURE_SMOOTHING_WINDOW,
    KP_LEFT_ANKLE, KP_RIGHT_ANKLE,
    KP_LEFT_HIP, KP_RIGHT_HIP,
    KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER,
)
from agriergo.interpretation.joint_angles import JointAngles


class PostureLabel(str, Enum):
    """Classified posture labels."""
    SITTING = "sitting"
    SQUATTING = "squatting"
    STANDING = "standing"
    BENDING = "bending"
    WALKING = "walking"
    UNKNOWN = "unknown"


@dataclass
class PostureResult:
    """Result of posture classification for a single frame."""
    label: PostureLabel
    raw_label: PostureLabel         # Before temporal smoothing
    confidence: float               # Heuristic confidence (0-1)
    trunk_flexion: Optional[float]
    hip_angle: Optional[float]
    knee_angle: Optional[float]
    displacement: Optional[float]


class PostureClassifier:
    """
    Rule-based posture classification from joint angles with ISO 11226 compliance.

    Classification logic (priority order):
    1. WALKING — if ankle displacement between frames exceeds threshold
    2. SQUATTING — if knee flexion < 100° and hips low near ankles
    3. BENDING — if trunk flexion > 30° from vertical (> 60° = severe stoop)
    4. SITTING — if hip angle < 120°
    5. STANDING — if trunk upright (< 15°) and hip angle > 160°
    6. UNKNOWN — insufficient keypoint data

    Temporal smoothing: confidence-weighted majority vote over a sliding window
    to ignore low-confidence frames caused by crop occlusion.
    """

    def __init__(self, smoothing_window: int = POSTURE_SMOOTHING_WINDOW):
        self.smoothing_window = smoothing_window
        # Per-worker label history: deque of (PostureLabel, confidence)
        self._history: Dict[int, deque] = {}
        # Per-worker previous ankle positions
        self._prev_ankles: Dict[int, Optional[np.ndarray]] = {}
        # Per-worker walking frame counter
        self._walk_counter: Dict[int, int] = {}

    def classify(
        self,
        worker_id: int,
        joint_angles: JointAngles,
        keypoints: np.ndarray,
        confidences: np.ndarray,
    ) -> PostureResult:
        """
        Classify posture for a worker in a single frame.

        Args:
            worker_id: Persistent worker ID.
            joint_angles: Computed joint angles for this frame.
            keypoints: Shape (17, 2) keypoint coordinates.
            confidences: Shape (17,) keypoint confidences.

        Returns:
            PostureResult with confidence-weighted smoothed label.
        """
        # Initialize per-worker state
        if worker_id not in self._history:
            self._history[worker_id] = deque(maxlen=self.smoothing_window)
            self._prev_ankles[worker_id] = None
            self._walk_counter[worker_id] = 0

        # Mean keypoint confidence for this detection
        valid_confs = confidences[confidences > 0.1]
        mean_conf = float(np.mean(valid_confs)) if len(valid_confs) > 0 else 0.5

        # Compute ankle displacement
        displacement = self._compute_displacement(worker_id, keypoints, confidences)

        # Apply classification rules
        raw_label, rule_confidence = self._apply_rules(
            joint_angles, displacement, worker_id, keypoints, confidences
        )
        
        frame_confidence = rule_confidence * mean_conf

        # Store in history for confidence-weighted temporal smoothing
        self._history[worker_id].append((raw_label, frame_confidence))
        smoothed_label = self._smooth(worker_id)

        return PostureResult(
            label=smoothed_label,
            raw_label=raw_label,
            confidence=round(frame_confidence, 2),
            trunk_flexion=joint_angles.trunk_flexion,
            hip_angle=joint_angles.avg_hip_angle,
            knee_angle=joint_angles.avg_knee_angle,
            displacement=displacement,
        )

    def _compute_displacement(
        self, worker_id: int, keypoints: np.ndarray, confidences: np.ndarray
    ) -> Optional[float]:
        """Compute ankle displacement between current and previous frame."""
        min_conf = 0.3
        l_valid = confidences[KP_LEFT_ANKLE] >= min_conf
        r_valid = confidences[KP_RIGHT_ANKLE] >= min_conf

        if not (l_valid or r_valid):
            return None

        if l_valid and r_valid:
            current = (keypoints[KP_LEFT_ANKLE] + keypoints[KP_RIGHT_ANKLE]) / 2
        elif l_valid:
            current = keypoints[KP_LEFT_ANKLE]
        else:
            current = keypoints[KP_RIGHT_ANKLE]

        prev = self._prev_ankles[worker_id]
        self._prev_ankles[worker_id] = current.copy()

        if prev is None:
            return None

        return float(np.linalg.norm(current - prev))

    def _apply_rules(
        self,
        angles: JointAngles,
        displacement: Optional[float],
        worker_id: int,
        keypoints: np.ndarray,
        confidences: np.ndarray,
    ) -> tuple:
        """
        Apply geometric classification rules:
        1. WALKING — Intermittent spatial displacement of ankle coordinates over continuous frames
        2. SQUATTING — Extreme flexion of knee joints (knee < 90°) with hips low to ankle line
        3. BENDING — Torso inclination drops (trunk flexion > 30° or hip < 135°) with knees largely straight (> 130°)
        4. SITTING — Hip & knee bent (~90° / hip < 120°, knee 70°-135°) with lower torso altitude
        5. STANDING — Hip, knee, and ankle align near vertical (knee > 150°, trunk upright < 20°, hip > 150°)
        """

        # Rule 1: WALKING — ankle displacement exceeds threshold over consecutive frames
        if displacement is not None and displacement > WALKING_DISPLACEMENT_THRESHOLD:
            self._walk_counter[worker_id] = self._walk_counter.get(worker_id, 0) + 1
            if self._walk_counter[worker_id] >= WALKING_MIN_CONSECUTIVE:
                return PostureLabel.WALKING, 0.88
        else:
            self._walk_counter[worker_id] = 0

        # Calculate Hip-to-Ankle vertical altitude if keypoints are available
        hip_y = None
        ankle_y = None
        shoulder_y = None
        if confidences[KP_LEFT_HIP] > 0.3 or confidences[KP_RIGHT_HIP] > 0.3:
            h_pts = [keypoints[i][1] for i in [KP_LEFT_HIP, KP_RIGHT_HIP] if confidences[i] > 0.3]
            hip_y = np.mean(h_pts)
        if confidences[KP_LEFT_ANKLE] > 0.3 or confidences[KP_RIGHT_ANKLE] > 0.3:
            a_pts = [keypoints[i][1] for i in [KP_LEFT_ANKLE, KP_RIGHT_ANKLE] if confidences[i] > 0.3]
            ankle_y = np.mean(a_pts)
        if confidences[KP_LEFT_SHOULDER] > 0.3 or confidences[KP_RIGHT_SHOULDER] > 0.3:
            s_pts = [keypoints[i][1] for i in [KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER] if confidences[i] > 0.3]
            shoulder_y = np.mean(s_pts)

        # Rule 2: SQUATTING — extreme flexion of knees (< 90°) with hips close to ankle line
        is_hips_low = False
        if hip_y is not None and ankle_y is not None and shoulder_y is not None:
            torso_h = abs(hip_y - shoulder_y)
            leg_h = abs(ankle_y - hip_y)
            if torso_h > 0 and leg_h < torso_h * 0.75:
                is_hips_low = True

        if angles.avg_knee_angle is not None and angles.avg_knee_angle < 95.0:
            if is_hips_low or (angles.avg_hip_angle is not None and angles.avg_hip_angle < 105.0):
                return PostureLabel.SQUATTING, 0.92

        # Rule 3: BENDING — torso inclination > 30° from vertical while knees largely straight (> 130°)
        if angles.trunk_flexion is not None and angles.trunk_flexion > TRUNK_FLEXION_BENDING:
            knee_straight = angles.avg_knee_angle is None or angles.avg_knee_angle >= 130.0
            if knee_straight:
                conf = min(0.96, 0.65 + (angles.trunk_flexion / 100.0))
                return PostureLabel.BENDING, conf

        # Rule 4: SITTING — hip & knee joints bent (hip < 120°, knee 70°-135°)
        if angles.avg_hip_angle is not None and angles.avg_hip_angle < HIP_ANGLE_SITTING:
            if angles.avg_knee_angle is not None and 70.0 <= angles.avg_knee_angle <= 135.0:
                return PostureLabel.SITTING, 0.85
            elif angles.trunk_flexion is not None and angles.trunk_flexion < 30.0:
                return PostureLabel.SITTING, 0.80

        # Rule 5: STANDING — hip, knee, and ankle align near vertical (knee > 150°, trunk < 20°, hip > 150°)
        if angles.trunk_flexion is not None and angles.trunk_flexion < TRUNK_FLEXION_UPRIGHT:
            if angles.avg_hip_angle is not None and angles.avg_hip_angle > 145.0:
                return PostureLabel.STANDING, 0.92
            if angles.avg_knee_angle is not None and angles.avg_knee_angle > 150.0:
                return PostureLabel.STANDING, 0.90

        # Default upright / bending fallback
        if angles.trunk_flexion is not None:
            if angles.trunk_flexion < TRUNK_FLEXION_BENDING:
                return PostureLabel.STANDING, 0.65
            else:
                return PostureLabel.BENDING, 0.70

        return PostureLabel.UNKNOWN, 0.0

    def _smooth(self, worker_id: int) -> PostureLabel:
        """Apply confidence-weighted majority vote over the history window."""
        history = self._history[worker_id]
        if not history:
            return PostureLabel.UNKNOWN

        # Sum confidence per label
        weights: Dict[PostureLabel, float] = {}
        for label, conf in history:
            weights[label] = weights.get(label, 0.0) + max(0.1, conf)

        return max(weights, key=weights.get)

    def reset(self, worker_id: Optional[int] = None):
        """Reset state for a specific worker or all workers."""
        if worker_id is not None:
            self._history.pop(worker_id, None)
            self._prev_ankles.pop(worker_id, None)
            self._walk_counter.pop(worker_id, None)
        else:
            self._history.clear()
            self._prev_ankles.clear()
            self._walk_counter.clear()
