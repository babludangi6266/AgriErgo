"""
Unit tests for Arm Postural Study (Shoulder Elevation & Elbow Flexion).
"""

import numpy as np
import pytest
from agriergo.interpretation.joint_angles import compute_joint_angles, JointAngles


def test_arm_angle_computation():
    # Setup keypoints for a person with right arm elevated horizontally (90 deg shoulder, 90 deg elbow)
    # COCO indices:
    # 5: L Shoulder, 6: R Shoulder, 7: L Elbow, 8: R Elbow, 9: L Wrist, 10: R Wrist
    # 11: L Hip, 12: R Hip, 13: L Knee, 14: R Knee, 15: L Ankle, 16: R Ankle
    keypoints = np.zeros((17, 2), dtype=float)
    confidences = np.ones(17, dtype=float)

    # Torso: Shoulders at y=100, Hips at y=200
    keypoints[5] = [90, 100]    # L Shoulder
    keypoints[6] = [110, 100]   # R Shoulder
    keypoints[11] = [95, 200]   # L Hip
    keypoints[12] = [105, 200]  # R Hip

    # Left arm: Hanging straight down at side (Elbow at y=150, Wrist at y=200)
    keypoints[7] = [90, 150]    # L Elbow
    keypoints[9] = [90, 200]    # L Wrist

    # Right arm: Raised horizontally to right (Elbow at x=160, y=100; Forearm pointing up to Wrist at x=160, y=50)
    keypoints[8] = [160, 100]   # R Elbow
    keypoints[10] = [160, 50]   # R Wrist

    angles = compute_joint_angles(keypoints, confidences)

    # Left shoulder elevation should be near 0 deg (hanging along torso)
    assert angles.left_shoulder_angle is not None
    assert abs(angles.left_shoulder_angle - 0.0) < 10.0

    # Right shoulder elevation should be ~90 deg (horizontal)
    assert angles.right_shoulder_angle is not None
    assert abs(angles.right_shoulder_angle - 90.0) < 10.0

    # Max shoulder angle should be ~90 deg and is_arm_elevated_45 should be True
    assert angles.max_shoulder_angle is not None
    assert angles.max_shoulder_angle >= 80.0
    assert angles.is_arm_elevated_45 is True

    # Left elbow should be ~180 deg (straight)
    assert angles.left_elbow_angle is not None
    assert abs(angles.left_elbow_angle - 180.0) < 5.0

    # Right elbow should be ~90 deg (bent upward)
    assert angles.right_elbow_angle is not None
    assert abs(angles.right_elbow_angle - 90.0) < 10.0
