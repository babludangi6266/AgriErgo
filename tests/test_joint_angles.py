"""
Unit tests for joint angle calculations.
"""

import pytest
import numpy as np
from agriergo.interpretation.joint_angles import (
    compute_angle,
    compute_angle_from_vertical,
    compute_joint_angles,
)


def test_compute_angle_right_angle():
    """Test 90 degree angle calculation."""
    p1 = np.array([0.0, 1.0])
    p2 = np.array([0.0, 0.0])
    p3 = np.array([1.0, 0.0])

    angle = compute_angle(p1, p2, p3)
    assert pytest.approx(angle, 0.1) == 90.0


def test_compute_angle_straight_line():
    """Test 180 degree angle calculation."""
    p1 = np.array([0.0, 1.0])
    p2 = np.array([0.0, 0.0])
    p3 = np.array([0.0, -1.0])

    angle = compute_angle(p1, p2, p3)
    assert pytest.approx(angle, 0.1) == 180.0


def test_compute_angle_from_vertical():
    """Test trunk flexion angle from vertical."""
    # Perfectly vertical segment (top above bottom)
    # Image coords: y increases downwards, so top has smaller y
    p_top = np.array([100.0, 50.0])
    p_bottom = np.array([100.0, 150.0])

    angle = compute_angle_from_vertical(p_top, p_bottom)
    assert pytest.approx(angle, abs=1e-1) == 0.0

    # 45 degree tilt forward
    p_top_bent = np.array([150.0, 100.0])
    p_bottom_bent = np.array([100.0, 150.0])
    angle_bent = compute_angle_from_vertical(p_top_bent, p_bottom_bent)
    assert pytest.approx(angle_bent, abs=1.0) == 45.0


def test_compute_joint_angles_structure():
    """Test compute_joint_angles with full keypoint array."""
    keypoints = np.zeros((17, 2))
    confidences = np.ones(17)

    # Set up synthetic person standing upright
    # 5: L-Shoulder, 6: R-Shoulder
    keypoints[5] = [90, 100]
    keypoints[6] = [110, 100]
    # 11: L-Hip, 12: R-Hip
    keypoints[11] = [90, 200]
    keypoints[12] = [110, 200]
    # 13: L-Knee, 14: R-Knee
    keypoints[13] = [90, 300]
    keypoints[14] = [110, 300]
    # 15: L-Ankle, 16: R-Ankle
    keypoints[15] = [90, 400]
    keypoints[16] = [110, 400]

    angles = compute_joint_angles(keypoints, confidences)
    assert angles.trunk_flexion is not None
    assert pytest.approx(angles.trunk_flexion, abs=1e-1) == 0.0
    assert angles.avg_hip_angle is not None
    assert pytest.approx(angles.avg_hip_angle, abs=1.0) == 180.0
