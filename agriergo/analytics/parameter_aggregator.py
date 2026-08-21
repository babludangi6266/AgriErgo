"""
Parameter Aggregator — Computes all 11 output parameters per worker.

Takes per-frame data (posture labels, joint angles, detected objects,
trajectories) and aggregates them into the 11 structured parameters
defined in the project requirements.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from agriergo.interpretation.posture_classifier import PostureLabel
from agriergo.interpretation.activity_segmenter import ActivityBout
from agriergo.interpretation.repetition_detector import RepetitionResult
from agriergo.interpretation.trip_counter import TripCountResult


@dataclass
class LoadInstance:
    """A detected instance of load carrying."""
    timestamp: float
    object_class: str
    weight_kg: Optional[float] = None   # Manual input — cannot derive from video
    duration: Optional[float] = None


@dataclass
class ToolUsageRecord:
    """A period when a specific tool/equipment was detected near a worker."""
    tool_name: str
    first_seen: float
    last_seen: float
    duration: float
    detection_count: int


@dataclass
class PostureSummary:
    """Summary of posture and ergonomic angle data for reporting."""
    dominant_posture: PostureLabel
    posture_distribution: Dict[str, float]   # label → percentage
    avg_trunk_flexion: Optional[float] = None
    max_trunk_flexion: Optional[float] = None
    avg_hip_angle: Optional[float] = None
    avg_knee_angle: Optional[float] = None
    angle_time_series: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class WorkerReport:
    """
    Complete report for a single worker containing all parameters.
    """
    worker_id: int
    total_tracked_time: float           # Total time worker was tracked (seconds)

    # Posture durations (seconds)
    sitting_duration: float = 0.0
    squatting_duration: float = 0.0     # Parameter 1b: Squatting duration
    standing_duration: float = 0.0
    bending_duration: float = 0.0
    severe_bending_duration: float = 0.0 # Bending > 60 degrees
    walking_duration: float = 0.0       # Parameter 4: Walking duration

    # Parameter 5: Load carried
    load_instances: List[LoadInstance] = field(default_factory=list)
    total_load_events: int = 0

    # Parameter 6: Repetitive movement frequency
    repetitive_movement: Optional[RepetitionResult] = None

    # Parameter 7: Number of trips
    trip_count_result: Optional[TripCountResult] = None

    # Parameter 8: Tools/equipment used
    tools_used: List[ToolUsageRecord] = field(default_factory=list)

    # Parameter 9: Posture summary
    posture_summary: Optional[PostureSummary] = None

    # Parameter 10: Continuous work duration
    longest_work_bout: float = 0.0     # Seconds
    avg_work_bout: float = 0.0         # Seconds
    work_bouts: List[ActivityBout] = field(default_factory=list)

    # Parameter 11: Rest duration
    total_rest_duration: float = 0.0    # Seconds
    rest_count: int = 0
    avg_rest_duration: float = 0.0      # Seconds
    rest_bouts: List[ActivityBout] = field(default_factory=list)

    # Activity timeline
    activity_bouts: List[ActivityBout] = field(default_factory=list)

    # Advanced Risk & Drudgery Metrics
    reba_score: Optional[int] = None
    reba_risk_level: Optional[str] = None
    rula_score: Optional[int] = None
    rula_action_level: Optional[str] = None
    drudgery_index: Optional[float] = None
    drudgery_category: Optional[str] = None
    drudgery_recommendations: List[str] = field(default_factory=list)
    fatigue_level: Optional[float] = None
    minute_fatigue_series: List[float] = field(default_factory=list)

    # NIOSH & Spinal Compression Metrics
    niosh_rwl_kg: Optional[float] = None
    niosh_lifting_index: Optional[float] = None
    l5s1_compression_n: Optional[float] = None
    niosh_risk_assessment: Optional[str] = None

    # Task Auto-Classification
    classified_task: Optional[str] = None
    task_confidence: Optional[float] = None
    task_hazard_profile: Optional[str] = None

    # Shift-Level Biomechanics & ISO 11226 Limits
    iso_11226_violated: bool = False
    iso_11226_message: Optional[str] = None
    shift_5min_posture_windows: List[Dict[str, Any]] = field(default_factory=list)


class ParameterAggregator:
    """
    Aggregates per-frame data into the 11 output parameters for each worker.
    """

    def aggregate(
        self,
        worker_id: int,
        activity_bouts: List[ActivityBout],
        repetition_result: Optional[RepetitionResult],
        trip_result: Optional[TripCountResult],
        tool_detections: Dict[str, List[float]],   # tool_name → list of timestamps
        load_detections: List[Dict[str, Any]],      # [{timestamp, object_class}, ...]
        trunk_flexions: List[Optional[float]],
        hip_angles: List[Optional[float]],
        knee_angles: Optional[List[Optional[float]]] = None,
        timestamps: Optional[List[float]] = None,
    ) -> WorkerReport:
        """
        Aggregate all data sources into a WorkerReport.

        Args:
            worker_id: Worker identifier.
            activity_bouts: Segmented activity bouts.
            repetition_result: Repetitive motion analysis result.
            trip_result: Trip counting result.
            tool_detections: Tool name → list of detection timestamps.
            load_detections: List of load detection events.
            trunk_flexions: Per-frame trunk flexion values.
            hip_angles: Per-frame hip angle values.
            knee_angles: Per-frame knee angle values.
            timestamps: Per-frame timestamps.

        Returns:
            Complete WorkerReport with all 11 parameters.
        """
        report = WorkerReport(worker_id=worker_id, total_tracked_time=0.0)

        # Calculate total tracked time
        if activity_bouts:
            report.total_tracked_time = round(
                activity_bouts[-1].end_time - activity_bouts[0].start_time, 2
            )

        # ── Parameters 1-4: Posture durations ──
        report.activity_bouts = activity_bouts
        for bout in activity_bouts:
            if bout.activity == PostureLabel.SITTING:
                report.sitting_duration += bout.duration
            elif bout.activity == PostureLabel.SQUATTING:
                report.squatting_duration += bout.duration
            elif bout.activity == PostureLabel.STANDING and not bout.is_rest:
                report.standing_duration += bout.duration
            elif bout.activity == PostureLabel.BENDING:
                report.bending_duration += bout.duration
            elif bout.activity == PostureLabel.WALKING:
                report.walking_duration += bout.duration

        report.sitting_duration = round(report.sitting_duration, 2)
        report.squatting_duration = round(report.squatting_duration, 2)
        report.standing_duration = round(report.standing_duration, 2)
        report.bending_duration = round(report.bending_duration, 2)
        report.walking_duration = round(report.walking_duration, 2)

        # ── Parameter 5: Load carried ──
        for det in load_detections:
            report.load_instances.append(LoadInstance(
                timestamp=det.get("timestamp", 0.0),
                object_class=det.get("object_class", "unknown"),
                weight_kg=det.get("weight_kg"),
            ))
        report.total_load_events = len(report.load_instances)

        # ── Parameter 6: Repetitive movement ──
        report.repetitive_movement = repetition_result

        # ── Parameter 7: Trips ──
        report.trip_count_result = trip_result

        # ── Parameter 8: Tools/equipment used ──
        report.tools_used = self._aggregate_tool_usage(tool_detections)

        # ── Parameter 9: Posture summary ──
        report.posture_summary = self._compute_posture_summary(
            activity_bouts, trunk_flexions, hip_angles, knee_angles, timestamps
        )

        # ── Parameters 10 & 11: Work and rest durations ──
        work_bouts = [b for b in activity_bouts if not b.is_rest]
        rest_bouts = [b for b in activity_bouts if b.is_rest]

        report.work_bouts = work_bouts
        report.rest_bouts = rest_bouts

        if work_bouts:
            work_durations = [b.duration for b in work_bouts]
            report.longest_work_bout = round(max(work_durations), 2)
            report.avg_work_bout = round(np.mean(work_durations), 2)

        if rest_bouts:
            rest_durations = [b.duration for b in rest_bouts]
            report.total_rest_duration = round(sum(rest_durations), 2)
            report.rest_count = len(rest_bouts)
            report.avg_rest_duration = round(np.mean(rest_durations), 2)

        return report

    def _aggregate_tool_usage(
        self, tool_detections: Dict[str, List[float]]
    ) -> List[ToolUsageRecord]:
        """Aggregate tool detection timestamps into usage records."""
        records = []
        for tool_name, timestamps in tool_detections.items():
            if not timestamps:
                continue
            timestamps.sort()
            records.append(ToolUsageRecord(
                tool_name=tool_name,
                first_seen=round(timestamps[0], 2),
                last_seen=round(timestamps[-1], 2),
                duration=round(timestamps[-1] - timestamps[0], 2),
                detection_count=len(timestamps),
            ))
        return records

    def _compute_posture_summary(
        self,
        bouts: List[ActivityBout],
        trunk_flexions: List[Optional[float]],
        hip_angles: List[Optional[float]],
        knee_angles: Optional[List[Optional[float]]] = None,
        timestamps: Optional[List[float]] = None,
    ) -> PostureSummary:
        """Compute posture distribution, ergonomic angle statistics, and time-series."""
        # Duration per posture
        posture_durations: Dict[str, float] = {}
        total_duration = 0.0
        for bout in bouts:
            label = bout.activity.value
            posture_durations[label] = posture_durations.get(label, 0.0) + bout.duration
            total_duration += bout.duration

        # Convert to percentages
        distribution = {}
        for label, dur in posture_durations.items():
            distribution[label] = round(
                (dur / total_duration * 100) if total_duration > 0 else 0, 1
            )

        # Find dominant posture
        dominant = max(posture_durations, key=posture_durations.get) if posture_durations else "unknown"

        # Trunk flexion stats
        valid_flexions = [f for f in trunk_flexions if f is not None]
        avg_flexion = round(np.mean(valid_flexions), 1) if valid_flexions else None
        max_flexion = round(max(valid_flexions), 1) if valid_flexions else None

        # Hip angle stats
        valid_hips = [a for a in hip_angles if a is not None]
        avg_hip = round(np.mean(valid_hips), 1) if valid_hips else None

        # Knee angle stats
        knee_list = knee_angles or []
        valid_knees = [k for k in knee_list if k is not None]
        avg_knee = round(np.mean(valid_knees), 1) if valid_knees else None

        # Angle Time Series for interactive visual plotting
        angle_series = []
        if timestamps and trunk_flexions:
            for idx, ts in enumerate(timestamps):
                tf = trunk_flexions[idx] if idx < len(trunk_flexions) else None
                ha = hip_angles[idx] if idx < len(hip_angles) else None
                ka = knee_list[idx] if idx < len(knee_list) else None
                if tf is not None or ha is not None or ka is not None:
                    angle_series.append({
                        "timestamp": round(ts, 2),
                        "trunk_flexion": round(tf, 1) if tf is not None else None,
                        "hip_angle": round(ha, 1) if ha is not None else None,
                        "knee_angle": round(ka, 1) if ka is not None else None,
                    })

        return PostureSummary(
            dominant_posture=PostureLabel(dominant),
            posture_distribution=distribution,
            avg_trunk_flexion=avg_flexion,
            max_trunk_flexion=max_flexion,
            avg_hip_angle=avg_hip,
            avg_knee_angle=avg_knee,
            angle_time_series=angle_series,
        )
