"""
Standardisation Engine — 1-Hour Activity Normalization.

Extrapolates and standardizes all tracked metrics (posture durations, repetitions,
trip counts, load carrying events, and cumulative ergonomic exposures) to a
standard 1-hour (3600 seconds) continuous activity baseline.

This enables fair comparison across videos of varying sample durations
(e.g., comparing a 2-minute video against a 25-minute field recording).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class Standardised1HrReport:
    """Standardised 1-Hour metrics for a single worker."""
    scaling_factor: float                     # 3600.0 / total_tracked_time
    total_standard_seconds: float = 3600.0    # 1 hour in seconds

    # Posture Durations per 1 Hour (seconds and formatted)
    sitting_seconds_1hr: float = 0.0
    sitting_formatted_1hr: str = "00:00:00"
    sitting_pct: float = 0.0

    squatting_seconds_1hr: float = 0.0
    squatting_formatted_1hr: str = "00:00:00"
    squatting_pct: float = 0.0

    standing_seconds_1hr: float = 0.0
    standing_formatted_1hr: str = "00:00:00"
    standing_pct: float = 0.0

    bending_seconds_1hr: float = 0.0
    bending_formatted_1hr: str = "00:00:00"
    bending_pct: float = 0.0

    severe_bending_seconds_1hr: float = 0.0
    severe_bending_formatted_1hr: str = "00:00:00"
    severe_bending_pct: float = 0.0

    walking_seconds_1hr: float = 0.0
    walking_formatted_1hr: str = "00:00:00"
    walking_pct: float = 0.0

    # Rest & Pause Durations per 1 Hour
    rest_seconds_1hr: float = 0.0
    rest_formatted_1hr: str = "00:00:00"
    rest_pct: float = 0.0
    rest_count_1hr: int = 0

    # Activity & Work Durations per 1 Hour
    active_work_seconds_1hr: float = 3600.0
    active_work_formatted_1hr: str = "01:00:00"

    # Repetitions & Load Events per 1 Hour
    repetitive_cycles_1hr: int = 0
    load_events_1hr: int = 0
    trips_1hr: int = 0

    # Arm Posture Exposure per 1 Hour
    arm_elevation_above_45_seconds_1hr: float = 0.0
    arm_elevation_above_90_seconds_1hr: float = 0.0


class HourlyStandardiser:
    """
    Standardises worker reports to a 1-hour activity session.
    """

    @staticmethod
    def standardise(
        total_tracked_time: float,
        sitting_duration: float,
        squatting_duration: float,
        standing_duration: float,
        bending_duration: float,
        severe_bending_duration: float,
        walking_duration: float,
        rest_duration: float,
        rest_count: int,
        load_events: int,
        trip_count: int,
        cycles_per_minute: Optional[float] = None,
        shoulder_above_45_pct: float = 0.0,
        shoulder_above_90_pct: float = 0.0,
    ) -> Standardised1HrReport:
        """
        Normalize all parameters to a 1-hour (3600s) activity window.
        """
        if total_tracked_time <= 0:
            scale = 1.0
        else:
            scale = 3600.0 / total_tracked_time

        # Calculate durations per hour capped at 3600s
        sit_1hr = min(3600.0, round(sitting_duration * scale, 1))
        squat_1hr = min(3600.0, round(squatting_duration * scale, 1))
        stand_1hr = min(3600.0, round(standing_duration * scale, 1))
        bend_1hr = min(3600.0, round(bending_duration * scale, 1))
        sev_bend_1hr = min(3600.0, round(severe_bending_duration * scale, 1))
        walk_1hr = min(3600.0, round(walking_duration * scale, 1))
        rest_1hr = min(3600.0, round(rest_duration * scale, 1))
        rest_cnt_1hr = int(round(rest_count * scale))

        active_work_1hr = max(0.0, round(3600.0 - rest_1hr, 1))

        # Repetitions per hour
        if cycles_per_minute is not None and cycles_per_minute > 0:
            # Active work minutes * CPM
            active_mins = active_work_1hr / 60.0
            rep_cycles_1hr = int(round(cycles_per_minute * active_mins))
        else:
            rep_cycles_1hr = 0

        # Load events & trips per hour
        load_1hr = int(round(load_events * scale))
        trips_1hr = int(round(trip_count * scale))

        # Arm elevation duration per hour
        arm_45_1hr = round((shoulder_above_45_pct / 100.0) * 3600.0, 1)
        arm_90_1hr = round((shoulder_above_90_pct / 100.0) * 3600.0, 1)

        # Percentages
        def _pct(v: float) -> float:
            return round((v / 3600.0) * 100.0, 1)

        return Standardised1HrReport(
            scaling_factor=round(scale, 3),
            total_standard_seconds=3600.0,
            sitting_seconds_1hr=sit_1hr,
            sitting_formatted_1hr=HourlyStandardiser._format_duration(sit_1hr),
            sitting_pct=_pct(sit_1hr),
            squatting_seconds_1hr=squat_1hr,
            squatting_formatted_1hr=HourlyStandardiser._format_duration(squat_1hr),
            squatting_pct=_pct(squat_1hr),
            standing_seconds_1hr=stand_1hr,
            standing_formatted_1hr=HourlyStandardiser._format_duration(stand_1hr),
            standing_pct=_pct(stand_1hr),
            bending_seconds_1hr=bend_1hr,
            bending_formatted_1hr=HourlyStandardiser._format_duration(bend_1hr),
            bending_pct=_pct(bend_1hr),
            severe_bending_seconds_1hr=sev_bend_1hr,
            severe_bending_formatted_1hr=HourlyStandardiser._format_duration(sev_bend_1hr),
            severe_bending_pct=_pct(sev_bend_1hr),
            walking_seconds_1hr=walk_1hr,
            walking_formatted_1hr=HourlyStandardiser._format_duration(walk_1hr),
            walking_pct=_pct(walk_1hr),
            rest_seconds_1hr=rest_1hr,
            rest_formatted_1hr=HourlyStandardiser._format_duration(rest_1hr),
            rest_pct=_pct(rest_1hr),
            rest_count_1hr=rest_cnt_1hr,
            active_work_seconds_1hr=active_work_1hr,
            active_work_formatted_1hr=HourlyStandardiser._format_duration(active_work_1hr),
            repetitive_cycles_1hr=rep_cycles_1hr,
            load_events_1hr=load_1hr,
            trips_1hr=trips_1hr,
            arm_elevation_above_45_seconds_1hr=arm_45_1hr,
            arm_elevation_above_90_seconds_1hr=arm_90_1hr,
        )

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format seconds into HH:MM:SS string."""
        s = max(0, int(round(seconds)))
        hours = s // 3600
        minutes = (s % 3600) // 60
        secs = s % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
