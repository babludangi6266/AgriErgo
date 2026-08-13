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
)
from agriergo.interpretation.joint_angles import JointAngles


class PostureLabel(str, Enum):
    """Classified posture labels."""
    SITTING = "sitting"
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
    displacement: Optional[float]


class PostureClassifier:
    """
    Rule-based posture classification from joint angles.

    Classification logic (priority order):
    1. WALKING — if ankle displacement between frames exceeds threshold
    2. BENDING — if trunk flexion > 30° from vertical
    3. SITTING — if hip angle < 120°
    4. STANDING — if trunk upright (< 15°) and hip angle > 160°
    5. UNKNOWN — insufficient keypoint data

    Temporal smoothing: majority-vote over a sliding window of N frames
    per worker to reduce flickering.
    """

    def __init__(self, smoothing_window: int = POSTURE_SMOOTHING_WINDOW):
        self.smoothing_window = smoothing_window
        # Per-worker label history for temporal smoothing
        self._history: Dict[int, deque] = {}
        # Per-worker previous ankle positions for displacement calculation
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
            PostureResult with smoothed label.
        """
        # Initialize per-worker state
        if worker_id not in self._history:
            self._history[worker_id] = deque(maxlen=self.smoothing_window)
            self._prev_ankles[worker_id] = None
            self._walk_counter[worker_id] = 0

        # ── Compute ankle displacement ──
        displacement = self._compute_displacement(worker_id, keypoints, confidences)

        # ── Apply classification rules ──
        raw_label, confidence = self._apply_rules(
            joint_angles, displacement, worker_id
        )

        # ── Temporal smoothing ──
        self._history[worker_id].append(raw_label)
        smoothed_label = self._smooth(worker_id)

        return PostureResult(
            label=smoothed_label,
            raw_label=raw_label,
            confidence=confidence,
            trunk_flexion=joint_angles.trunk_flexion,
            hip_angle=joint_angles.avg_hip_angle,
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

        # Use mid-ankle if both available, otherwise whichever is valid
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
    ) -> tuple:
        """Apply rule-based classification. Returns (label, confidence)."""

        # Rule 1: WALKING — ankle displacement exceeds threshold
        if displacement is not None and displacement > WALKING_DISPLACEMENT_THRESHOLD:
            self._walk_counter[worker_id] = self._walk_counter.get(worker_id, 0) + 1
            if self._walk_counter[worker_id] >= WALKING_MIN_CONSECUTIVE:
                return PostureLabel.WALKING, 0.8
        else:
            self._walk_counter[worker_id] = 0

        # Rule 2: BENDING — trunk flexion > threshold
        if angles.trunk_flexion is not None:
            if angles.trunk_flexion > TRUNK_FLEXION_BENDING:
                return PostureLabel.BENDING, min(0.9, angles.trunk_flexion / 60.0)

        # Rule 3: SITTING — hip angle < threshold
        if angles.avg_hip_angle is not None:
            if angles.avg_hip_angle < HIP_ANGLE_SITTING:
                return PostureLabel.SITTING, 0.75

        # Rule 4: STANDING — trunk upright + hip extended
        if angles.trunk_flexion is not None and angles.avg_hip_angle is not None:
            if (angles.trunk_flexion < TRUNK_FLEXION_UPRIGHT and
                    angles.avg_hip_angle > HIP_ANGLE_STANDING):
                return PostureLabel.STANDING, 0.85

        # If trunk is roughly upright but hip data is missing → likely standing
        if angles.trunk_flexion is not None and angles.trunk_flexion < TRUNK_FLEXION_BENDING:
            return PostureLabel.STANDING, 0.5

        return PostureLabel.UNKNOWN, 0.0

    def _smooth(self, worker_id: int) -> PostureLabel:
        """Apply majority-vote smoothing over the label history window."""
        history = self._history[worker_id]
        if not history:
            return PostureLabel.UNKNOWN

        # Count occurrences
        counts: Dict[PostureLabel, int] = {}
        for label in history:
            counts[label] = counts.get(label, 0) + 1

        # Return the most common label
        return max(counts, key=counts.get)

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
