"""
Joint Angle Computation — Calculates anatomically meaningful angles from keypoints.

Computes trunk flexion, hip angles, knee angles, elbow angles, neck angles,
and shoulder elevation from COCO-format keypoints for use in posture
classification and REBA scoring.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    KP_NOSE, KP_LEFT_EAR, KP_RIGHT_EAR,
    KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER,
    KP_LEFT_ELBOW, KP_RIGHT_ELBOW,
    KP_LEFT_WRIST, KP_RIGHT_WRIST,
    KP_LEFT_HIP, KP_RIGHT_HIP,
    KP_LEFT_KNEE, KP_RIGHT_KNEE,
    KP_LEFT_ANKLE, KP_RIGHT_ANKLE,
    POSE_CONFIDENCE,
)


@dataclass
class JointAngles:
    """All computed joint angles for a single person in a single frame."""
    # Trunk
    trunk_flexion: Optional[float] = None       # Degrees from vertical

    # Hips (shoulder-hip-knee)
    left_hip_angle: Optional[float] = None
    right_hip_angle: Optional[float] = None
    avg_hip_angle: Optional[float] = None

    # Knees (hip-knee-ankle)
    left_knee_angle: Optional[float] = None
    right_knee_angle: Optional[float] = None
    avg_knee_angle: Optional[float] = None

    # Elbows (shoulder-elbow-wrist)
    left_elbow_angle: Optional[float] = None
    right_elbow_angle: Optional[float] = None
    avg_elbow_angle: Optional[float] = None

    # Neck (nose/ear → shoulder vs vertical)
    neck_flexion: Optional[float] = None

    # Shoulder elevation (Upper arm relative to torso / vertical)
    left_shoulder_angle: Optional[float] = None   # Angle of upper arm elevation from neutral torso (deg)
    right_shoulder_angle: Optional[float] = None
    avg_shoulder_angle: Optional[float] = None
    max_shoulder_angle: Optional[float] = None
    is_arm_above_shoulder: bool = False           # Elevation > 90° (high ergonomic hazard)
    is_arm_elevated_45: bool = False              # Elevation > 45° (moderate ergonomic hazard)

    # Wrist deviation (angle at wrist)
    left_wrist_angle: Optional[float] = None
    right_wrist_angle: Optional[float] = None


def compute_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Compute the angle at p2 formed by vectors p2→p1 and p2→p3.

    Args:
        p1, p2, p3: 2D points as numpy arrays of shape (2,).

    Returns:
        Angle in degrees (0–180).
    """
    v1 = p1 - p2
    v2 = p3 - p2

    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))

    return float(angle)


def compute_angle_from_vertical(p_top: np.ndarray, p_bottom: np.ndarray) -> float:
    """
    Compute the angle of the vector (p_bottom → p_top) relative to the vertical (upward).

    A perfectly upright segment returns ~0°. Bending forward gives positive angles.

    Args:
        p_top: Upper point (e.g., mid-shoulder).
        p_bottom: Lower point (e.g., mid-hip).

    Returns:
        Angle in degrees from vertical (0–180).
    """
    # Vertical reference: straight up (0, -1) in image coords (y increases downward)
    segment = p_top - p_bottom  # Vector from bottom to top
    vertical = np.array([0.0, -1.0])  # Upward in image coordinates

    cos_angle = np.dot(segment, vertical) / (np.linalg.norm(segment) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))

    return float(angle)


def _midpoint(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Compute midpoint of two 2D points."""
    return (p1 + p2) / 2.0


def _is_valid(keypoints: np.ndarray, confidences: np.ndarray,
              *indices: int, min_conf: float = POSE_CONFIDENCE) -> bool:
    """Check if all specified keypoints have sufficient confidence."""
    for idx in indices:
        if confidences[idx] < min_conf:
            return False
        if np.allclose(keypoints[idx], 0.0):
            return False
    return True


def compute_joint_angles(
    keypoints: np.ndarray,
    confidences: np.ndarray,
    min_conf: float = POSE_CONFIDENCE,
) -> JointAngles:
    """
    Compute all joint angles from COCO-format keypoints.

    Args:
        keypoints: Shape (17, 2) — x, y pixel coordinates.
        confidences: Shape (17,) — per-keypoint confidence.
        min_conf: Minimum confidence to consider a keypoint valid.

    Returns:
        JointAngles dataclass with all computed angles.
    """
    angles = JointAngles()
    kp = keypoints
    conf = confidences

    # ── Trunk Flexion ──
    # Angle of (mid-shoulder → mid-hip) vector from vertical
    if _is_valid(kp, conf, KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER,
                 KP_LEFT_HIP, KP_RIGHT_HIP, min_conf=min_conf):
        mid_shoulder = _midpoint(kp[KP_LEFT_SHOULDER], kp[KP_RIGHT_SHOULDER])
        mid_hip = _midpoint(kp[KP_LEFT_HIP], kp[KP_RIGHT_HIP])
        angles.trunk_flexion = compute_angle_from_vertical(mid_shoulder, mid_hip)

    # ── Hip Angles (shoulder-hip-knee) ──
    if _is_valid(kp, conf, KP_LEFT_SHOULDER, KP_LEFT_HIP, KP_LEFT_KNEE, min_conf=min_conf):
        angles.left_hip_angle = compute_angle(
            kp[KP_LEFT_SHOULDER], kp[KP_LEFT_HIP], kp[KP_LEFT_KNEE]
        )

    if _is_valid(kp, conf, KP_RIGHT_SHOULDER, KP_RIGHT_HIP, KP_RIGHT_KNEE, min_conf=min_conf):
        angles.right_hip_angle = compute_angle(
            kp[KP_RIGHT_SHOULDER], kp[KP_RIGHT_HIP], kp[KP_RIGHT_KNEE]
        )

    # Average hip angle
    hip_vals = [v for v in [angles.left_hip_angle, angles.right_hip_angle] if v is not None]
    if hip_vals:
        angles.avg_hip_angle = sum(hip_vals) / len(hip_vals)

    # ── Knee Angles (hip-knee-ankle) ──
    if _is_valid(kp, conf, KP_LEFT_HIP, KP_LEFT_KNEE, KP_LEFT_ANKLE, min_conf=min_conf):
        angles.left_knee_angle = compute_angle(
            kp[KP_LEFT_HIP], kp[KP_LEFT_KNEE], kp[KP_LEFT_ANKLE]
        )

    if _is_valid(kp, conf, KP_RIGHT_HIP, KP_RIGHT_KNEE, KP_RIGHT_ANKLE, min_conf=min_conf):
        angles.right_knee_angle = compute_angle(
            kp[KP_RIGHT_HIP], kp[KP_RIGHT_KNEE], kp[KP_RIGHT_ANKLE]
        )

    knee_vals = [v for v in [angles.left_knee_angle, angles.right_knee_angle] if v is not None]
    if knee_vals:
        angles.avg_knee_angle = sum(knee_vals) / len(knee_vals)

    # ── Elbow Angles (shoulder-elbow-wrist via vector cosine rule) ──
    if _is_valid(kp, conf, KP_LEFT_SHOULDER, KP_LEFT_ELBOW, KP_LEFT_WRIST, min_conf=min_conf):
        angles.left_elbow_angle = compute_angle(
            kp[KP_LEFT_SHOULDER], kp[KP_LEFT_ELBOW], kp[KP_LEFT_WRIST]
        )

    if _is_valid(kp, conf, KP_RIGHT_SHOULDER, KP_RIGHT_ELBOW, KP_RIGHT_WRIST, min_conf=min_conf):
        angles.right_elbow_angle = compute_angle(
            kp[KP_RIGHT_SHOULDER], kp[KP_RIGHT_ELBOW], kp[KP_RIGHT_WRIST]
        )

    elbow_vals = [v for v in [angles.left_elbow_angle, angles.right_elbow_angle] if v is not None]
    if elbow_vals:
        angles.avg_elbow_angle = sum(elbow_vals) / len(elbow_vals)

    # ── Neck Flexion ──
    # Angle from vertical of (nose → mid-shoulder) vector
    if _is_valid(kp, conf, KP_NOSE, KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER, min_conf=min_conf):
        mid_shoulder = _midpoint(kp[KP_LEFT_SHOULDER], kp[KP_RIGHT_SHOULDER])
        angles.neck_flexion = compute_angle_from_vertical(kp[KP_NOSE], mid_shoulder)

    # ── Shoulder / Upper Arm Elevation Angles (vector dot product relative to torso) ──
    # Angle of upper arm relative to torso line (hip-shoulder-elbow)
    if _is_valid(kp, conf, KP_LEFT_SHOULDER, KP_LEFT_ELBOW,
                 KP_LEFT_HIP, min_conf=min_conf):
        # Angle at shoulder: hip-shoulder-elbow (0° hanging along torso, 90° horizontal, 180° overhead)
        angles.left_shoulder_angle = compute_angle(
            kp[KP_LEFT_HIP], kp[KP_LEFT_SHOULDER], kp[KP_LEFT_ELBOW]
        )

    if _is_valid(kp, conf, KP_RIGHT_SHOULDER, KP_RIGHT_ELBOW,
                 KP_RIGHT_HIP, min_conf=min_conf):
        angles.right_shoulder_angle = compute_angle(
            kp[KP_RIGHT_HIP], kp[KP_RIGHT_SHOULDER], kp[KP_RIGHT_ELBOW]
        )

    shoulder_vals = [v for v in [angles.left_shoulder_angle, angles.right_shoulder_angle] if v is not None]
    if shoulder_vals:
        angles.avg_shoulder_angle = sum(shoulder_vals) / len(shoulder_vals)
        angles.max_shoulder_angle = max(shoulder_vals)
        angles.is_arm_above_shoulder = any(v >= 90.0 for v in shoulder_vals)
        angles.is_arm_elevated_45 = any(v >= 45.0 for v in shoulder_vals)

    # ── Wrist Angles ──
    if _is_valid(kp, conf, KP_LEFT_ELBOW, KP_LEFT_WRIST, min_conf=min_conf):
        # Simplified: angle of wrist from forearm line relative to vertical
        wrist_vec = kp[KP_LEFT_WRIST] - kp[KP_LEFT_ELBOW]
        vertical = np.array([0.0, -1.0])
        cos_a = np.dot(wrist_vec, vertical) / (np.linalg.norm(wrist_vec) + 1e-8)
        cos_a = np.clip(cos_a, -1.0, 1.0)
        angles.left_wrist_angle = float(np.degrees(np.arccos(cos_a)))

    if _is_valid(kp, conf, KP_RIGHT_ELBOW, KP_RIGHT_WRIST, min_conf=min_conf):
        wrist_vec = kp[KP_RIGHT_WRIST] - kp[KP_RIGHT_ELBOW]
        vertical = np.array([0.0, -1.0])
        cos_a = np.dot(wrist_vec, vertical) / (np.linalg.norm(wrist_vec) + 1e-8)
        cos_a = np.clip(cos_a, -1.0, 1.0)
        angles.right_wrist_angle = float(np.degrees(np.arccos(cos_a)))

    return angles
