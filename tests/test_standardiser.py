"""
Unit tests for HourlyStandardiser and 1-Hour Activity Normalization.
"""

import pytest
from agriergo.analytics.standardiser import HourlyStandardiser, Standardised1HrReport


def test_standardiser_basic_scaling():
    # 60-second video sample -> 60x scaling factor for 1 hour (3600s)
    report = HourlyStandardiser.standardise(
        total_tracked_time=60.0,
        sitting_duration=10.0,
        squatting_duration=10.0,
        standing_duration=20.0,
        bending_duration=15.0,
        severe_bending_duration=5.0,
        walking_duration=5.0,
        rest_duration=6.0,
        rest_count=1,
        load_events=2,
        trip_count=3,
        cycles_per_minute=25.0,
        shoulder_above_45_pct=30.0,
        shoulder_above_90_pct=10.0,
    )

    assert report.scaling_factor == 60.0
    assert report.total_standard_seconds == 3600.0
    assert report.sitting_seconds_1hr == 600.0
    assert report.sitting_formatted_1hr == "00:10:00"
    assert report.standing_seconds_1hr == 1200.0
    assert report.bending_seconds_1hr == 900.0
    assert report.rest_seconds_1hr == 3600.0 * (6.0 / 60.0)  # 360s = 00:06:00
    assert report.rest_formatted_1hr == "00:06:00"
    assert report.load_events_1hr == 120
    assert report.trips_1hr == 180
    assert report.repetitive_cycles_1hr > 0


def test_standardiser_zero_duration():
    report = HourlyStandardiser.standardise(
        total_tracked_time=0.0,
        sitting_duration=0.0,
        squatting_duration=0.0,
        standing_duration=0.0,
        bending_duration=0.0,
        severe_bending_duration=0.0,
        walking_duration=0.0,
        rest_duration=0.0,
        rest_count=0,
        load_events=0,
        trip_count=0,
    )
    assert report.scaling_factor == 1.0
    assert report.sitting_seconds_1hr == 0.0
