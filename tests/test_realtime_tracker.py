"""
Tests for Real-Time Body Angle & Posture Tracker (agriergo.perception.realtime_tracker).
"""

import numpy as np
import pytest

from agriergo.perception.realtime_tracker import (
    RealTimePostureTracker,
    RealTimeFrameResult,
    WorkerLiveTelemetry,
)
from agriergo.interpretation.posture_classifier import PostureLabel
from agriergo.interpretation.joint_angles import JointAngles


def test_worker_live_telemetry_defaults():
    """Test WorkerLiveTelemetry default values and properties."""
    telem = WorkerLiveTelemetry(
        worker_id=1,
        posture=PostureLabel.BENDING,
        trunk_angle=62.5,
        reba_score=8,
        reba_risk_level="High",
        is_severe_stoop=True,
    )
    assert telem.worker_id == 1
    assert telem.posture == PostureLabel.BENDING
    assert telem.trunk_angle == 62.5
    assert telem.is_severe_stoop is True
    assert telem.reba_score == 8


def test_realtime_frame_result_structure():
    """Test RealTimeFrameResult fields."""
    dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    res = RealTimeFrameResult(
        annotated_frame=dummy_frame,
        fps=28.5,
        timestamp=1000.0,
        workers=[],
        primary_telemetry=None,
        alerts=["Severe stoop"],
    )
    assert res.fps == 28.5
    assert len(res.alerts) == 1
    assert res.annotated_frame.shape == (240, 320, 3)


def test_angle_smoothing_ema():
    """Test Exponential Moving Average angle smoothing logic."""
    tracker = RealTimePostureTracker(smoothing_alpha=0.5)

    raw1 = JointAngles(trunk_flexion=40.0, avg_knee_angle=170.0)
    smoothed1 = tracker._smooth_angles(worker_id=1, raw=raw1)
    assert smoothed1["trunk_flexion"] == 40.0

    # Second frame with angle change to 60.0
    # Expected EMA = 0.5 * 60.0 + 0.5 * 40.0 = 50.0
    raw2 = JointAngles(trunk_flexion=60.0, avg_knee_angle=170.0)
    smoothed2 = tracker._smooth_angles(worker_id=1, raw=raw2)
    assert abs(smoothed2["trunk_flexion"] - 50.0) < 1e-4


def test_posture_duration_accumulation():
    """Test posture duration tracking."""
    tracker = RealTimePostureTracker()
    tracker._update_durations(worker_id=1, posture=PostureLabel.BENDING, dt=1.0, now=10.0)
    tracker._update_durations(worker_id=1, posture=PostureLabel.BENDING, dt=2.0, now=12.0)

    durations = tracker._posture_durations[1]
    assert durations["bending"] == 3.0


def test_tracker_reset():
    """Test resetting internal tracker state."""
    tracker = RealTimePostureTracker()
    raw = JointAngles(trunk_flexion=45.0)
    tracker._smooth_angles(worker_id=1, raw=raw)
    assert len(tracker._smoothed_angles) == 1

    tracker.reset()
    assert len(tracker._smoothed_angles) == 0
    assert tracker.frame_count == 0
