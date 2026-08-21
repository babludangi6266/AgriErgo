"""
Unit tests for Agricultural Task Auto-Classifier.
"""

import pytest
from agriergo.interpretation.task_classifier import TaskClassifier


def test_task_classifier_weeding_and_seed_prep():
    classifier = TaskClassifier()
    res = classifier.classify_task(
        posture_distribution={"bending": 45.0, "squatting": 25.0},
        cycles_per_minute=32.0,
        repetitive_joint="elbow",
        detected_tools=[],
        total_load_events=0,
    )
    assert res.primary_task in ["Seed Rhizome Preparation & Treatment", "Sorting & Grading", "Sowing & Rhizome Planting", "Land & Bed Preparation (Hoeing / Ridging)"]
    assert res.confidence >= 0.8


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


def test_task_classifier_turmeric_activities():
    classifier = TaskClassifier()
    # Land prep / hoeing
    res_hoeing = classifier.classify_task(
        posture_distribution={"bending": 20.0, "walking": 50.0},
        cycles_per_minute=28.0,
        repetitive_joint="elbow",
        detected_tools=[],
        total_load_events=0,
    )
    assert res_hoeing.primary_task == "Land & Bed Preparation (Hoeing / Ridging)"

    # Cleaning & Rhizome separation
    res_clean = classifier.classify_task(
        posture_distribution={"squatting": 50.0, "sitting": 20.0},
        cycles_per_minute=45.0,
        repetitive_joint="wrist",
        detected_tools=[],
        total_load_events=0,
    )
    assert res_clean.primary_task == "Cleaning & Rhizome Separation"
