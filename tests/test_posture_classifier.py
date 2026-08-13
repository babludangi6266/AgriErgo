"""
Unit tests for posture classification rules.
"""

import pytest
import numpy as np
from agriergo.interpretation.joint_angles import JointAngles
from agriergo.interpretation.posture_classifier import PostureClassifier, PostureLabel


def test_posture_bending():
    """Test bending classification when trunk flexion > 30 degrees."""
    classifier = PostureClassifier(smoothing_window=1)
    angles = JointAngles(trunk_flexion=45.0, avg_hip_angle=160.0)

    keypoints = np.zeros((17, 2))
    confidences = np.zeros(17)

    res = classifier.classify(1, angles, keypoints, confidences)
    assert res.label == PostureLabel.BENDING


def test_posture_sitting():
    """Test sitting classification when hip angle < 120 degrees."""
    classifier = PostureClassifier(smoothing_window=1)
    angles = JointAngles(trunk_flexion=10.0, avg_hip_angle=90.0)

    keypoints = np.zeros((17, 2))
    confidences = np.zeros(17)

    res = classifier.classify(1, angles, keypoints, confidences)
    assert res.label == PostureLabel.SITTING


def test_posture_squatting():
    """Test squatting classification when knee angle < 100 degrees and hip < 110 degrees."""
    classifier = PostureClassifier(smoothing_window=1)
    angles = JointAngles(trunk_flexion=15.0, avg_hip_angle=90.0, avg_knee_angle=70.0)

    keypoints = np.zeros((17, 2))
    confidences = np.ones(17)

    res = classifier.classify(1, angles, keypoints, confidences)
    assert res.label == PostureLabel.SQUATTING


def test_posture_standing():
    """Test standing classification when upright and extended hips."""
    classifier = PostureClassifier(smoothing_window=1)
    angles = JointAngles(trunk_flexion=5.0, avg_hip_angle=175.0)

    keypoints = np.zeros((17, 2))
    confidences = np.zeros(17)

    res = classifier.classify(1, angles, keypoints, confidences)
    assert res.label == PostureLabel.STANDING


def test_posture_smoothing():
    """Test temporal smoothing window majority vote."""
    classifier = PostureClassifier(smoothing_window=3)
    keypoints = np.zeros((17, 2))
    confidences = np.zeros(17)

    angles_stand = JointAngles(trunk_flexion=5.0, avg_hip_angle=175.0)
    angles_bend = JointAngles(trunk_flexion=45.0, avg_hip_angle=160.0)

    # Frame 1: Standing
    res1 = classifier.classify(1, angles_stand, keypoints, confidences)
    assert res1.label == PostureLabel.STANDING

    # Frame 2: Standing
    res2 = classifier.classify(1, angles_stand, keypoints, confidences)
    assert res2.label == PostureLabel.STANDING

    # Frame 3: Single flicker frame of Bending -> should stay Standing due to majority (2 standing vs 1 bending)
    res3 = classifier.classify(1, angles_bend, keypoints, confidences)
    assert res3.label == PostureLabel.STANDING
