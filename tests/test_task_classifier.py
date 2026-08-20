"""
Unit tests for Agricultural Task Auto-Classifier.
"""

import pytest
from agriergo.interpretation.task_classifier import TaskClassifier


def test_task_classifier_weeding():
    classifier = TaskClassifier()
    res = classifier.classify_task(
        posture_distribution={"bending": 45.0, "squatting": 25.0},
        cycles_per_minute=30.0,
        repetitive_joint="elbow",
        detected_tools=[],
        total_load_events=0,
    )
    assert res.primary_task == "Manual Weeding / Ground Harvesting"
    assert res.confidence > 0.7


def test_task_classifier_transport():
    classifier = TaskClassifier()
    res = classifier.classify_task(
        posture_distribution={"walking": 40.0, "standing": 40.0},
        cycles_per_minute=0.0,
        repetitive_joint=None,
        detected_tools=["backpack"],
        total_load_events=3,
    )
    assert res.primary_task == "Crop Transport / Load Carrying"
    assert res.confidence > 0.8
