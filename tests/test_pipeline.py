"""
Integration tests for the AgriErgo Pipeline.
"""

import pytest
import numpy as np
from agriergo.interpretation.activity_segmenter import ActivitySegmenter, FrameRecord, PostureLabel
from agriergo.interpretation.repetition_detector import RepetitionDetector
from agriergo.analytics.parameter_aggregator import ParameterAggregator
from agriergo.analytics.ergonomic_scorer import ErgonomicScorer, JointAngles


def test_activity_segmenter_bouts():
    """Test activity bout segmentation and filtering."""
    segmenter = ActivitySegmenter(min_bout_duration=1.0)
    records = [
        FrameRecord(frame_idx=i, timestamp=i*0.2, posture=PostureLabel.STANDING)
        for i in range(10)  # 2.0s standing
    ] + [
        FrameRecord(frame_idx=i+10, timestamp=(i+10)*0.2, posture=PostureLabel.BENDING)
        for i in range(15) # 3.0s bending
    ]

    bouts = segmenter.segment(worker_id=1, frame_records=records)
    assert len(bouts) == 2
    assert bouts[0].activity == PostureLabel.STANDING
    assert bouts[1].activity == PostureLabel.BENDING


def test_repetition_detector_synthetic_sine():
    """Test repetition detection on synthetic periodic angle signal."""
    detector = RepetitionDetector()
    fps = 10.0
    t = np.linspace(0, 10, int(10 * fps))
    # 1.0 Hz sine wave
    signal = 90.0 + 20.0 * np.sin(2 * np.pi * 1.0 * t)

    res = detector.detect_frequency(signal, sample_fps=fps)
    assert res.is_repetitive
    assert res.frequency_hz is not None
    assert pytest.approx(res.frequency_hz, 0.2) == 1.0


def test_ergonomic_scorer_reba():
    """Test REBA score computation."""
    scorer = ErgonomicScorer()
    angles = JointAngles(
        trunk_flexion=65.0,   # >60 deg flexion -> trunk score 4
        neck_flexion=25.0,    # >20 deg flexion -> neck score 2
        avg_knee_angle=170.0,
        left_shoulder_angle=80.0,
        left_elbow_angle=90.0,
    )

    score = scorer.score_frame(angles, is_static=True, is_repetitive=True)
    assert score.final_score >= 4
    assert score.risk_level in ["Medium", "High", "Very High"]
