"""
Unit tests for RULA Ergonomic Scoring Engine.
"""

import pytest
from agriergo.interpretation.joint_angles import JointAngles
from agriergo.analytics.rula_scorer import RULAScorer


def test_rula_scoring_neutral():
    """Test RULA scoring for neutral standing posture."""
    scorer = RULAScorer()
    angles = JointAngles(
        trunk_flexion=0.0,
        neck_flexion=5.0,
        avg_knee_angle=180.0,
        left_shoulder_angle=10.0,
        left_elbow_angle=90.0,
        left_wrist_angle=90.0,
    )
    score = scorer.score_frame(angles)
    assert score.final_score <= 3
    assert "Acceptable" in score.action_level or "Investigate" in score.action_level


def test_rula_scoring_severe():
    """Test RULA scoring for severe upper arm and stooped posture."""
    scorer = RULAScorer()
    angles = JointAngles(
        trunk_flexion=65.0,
        neck_flexion=35.0,
        avg_knee_angle=80.0,
        left_shoulder_angle=95.0,
        left_elbow_angle=120.0,
        left_wrist_angle=40.0,
    )
    score = scorer.score_frame(angles, is_repetitive=True, load_force=2)
    assert score.final_score >= 5
    assert "Change" in score.action_level
