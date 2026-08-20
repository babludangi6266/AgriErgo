"""
Unit tests for Agricultural Drudgery Index calculator.
"""

import pytest
from agriergo.analytics.drudgery_index import DrudgeryCalculator


def test_drudgery_calculator():
    calc = DrudgeryCalculator()
    res = calc.calculate(
        reba_score=8.0,
        rula_score=5.0,
        posture_distribution={"bending": 40.0, "squatting": 20.0},
        cycles_per_minute=45.0,
        total_tracked_seconds=600.0,
        longest_work_bout_seconds=300.0,
        total_rest_seconds=30.0,
        load_events_count=2,
    )

    assert res.drudgery_index > 0.0
    assert res.drudgery_index <= 100.0
    assert res.drudgery_category in ["Moderate Drudgery", "High Drudgery", "Severe Drudgery"]
    assert len(res.recommendations) > 0
