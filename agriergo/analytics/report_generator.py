"""
Report Generator — Produces structured JSON and CSV reports.

Serializes WorkerReport data into human-readable and machine-readable
formats for download and further analysis.
"""

import json
import csv
import io
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import asdict

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import numpy as np
from agriergo.analytics.parameter_aggregator import WorkerReport
from agriergo.perception.video_processor import VideoMetadata


def _to_python_native(obj):
    """Recursively convert numpy data types and objects into standard Python primitives."""
    if hasattr(obj, 'item') and callable(getattr(obj, 'item')) and not isinstance(obj, (list, tuple, dict, str, np.ndarray)):
        try:
            obj = obj.item()
        except Exception:
            pass

    if isinstance(obj, (np.bool_, bool)) or type(obj).__name__ in ("bool_", "bool"):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [_to_python_native(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _to_python_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python_native(x) for x in obj]
    return obj


class ReportGenerator:
    """
    Generates structured reports from WorkerReport data.

    Output formats:
    - JSON: Full structured data with all parameters
    - CSV: Tabular summary (one row per worker)
    """

    def generate_json(
        self,
        worker_reports: List[WorkerReport],
        video_metadata: VideoMetadata,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a structured JSON report.

        Args:
            worker_reports: List of reports for all tracked workers.
            video_metadata: Source video metadata.
            output_path: If provided, write JSON to this file.

        Returns:
            Report as a Python dictionary.
        """
        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "platform": "AgriErgo v0.1.0",
                "report_type": "ergonomic_assessment",
            },
            "video_info": {
                "filename": video_metadata.filename,
                "duration_seconds": float(video_metadata.duration_seconds),
                "resolution": f"{video_metadata.width}x{video_metadata.height}",
                "fps": float(video_metadata.fps),
                "total_frames": int(video_metadata.total_frames),
            },
            "summary": {
                "total_workers_detected": len(worker_reports),
                "video_duration_formatted": self._format_duration(
                    video_metadata.duration_seconds
                ),
            },
            "workers": [],
        }

        for wr in worker_reports:
            worker_data = {
                "worker_id": int(wr.worker_id),
                "total_tracked_time": float(wr.total_tracked_time),
                "total_tracked_time_formatted": self._format_duration(
                    wr.total_tracked_time
                ),
                "parameters": {
                    "1_sitting": {
                        "duration_seconds": float(wr.sitting_duration),
                        "duration_formatted": self._format_duration(wr.sitting_duration),
                        "percentage": float(self._pct(wr.sitting_duration, wr.total_tracked_time)),
                    },
                    "1b_squatting": {
                        "duration_seconds": float(getattr(wr, 'squatting_duration', 0.0)),
                        "duration_formatted": self._format_duration(getattr(wr, 'squatting_duration', 0.0)),
                        "percentage": float(self._pct(getattr(wr, 'squatting_duration', 0.0), wr.total_tracked_time)),
                    },
                    "2_standing": {
                        "duration_seconds": float(wr.standing_duration),
                        "duration_formatted": self._format_duration(wr.standing_duration),
                        "percentage": float(self._pct(wr.standing_duration, wr.total_tracked_time)),
                    },
                    "3_bending": {
                        "duration_seconds": float(wr.bending_duration),
                        "duration_formatted": self._format_duration(wr.bending_duration),
                        "percentage": float(self._pct(wr.bending_duration, wr.total_tracked_time)),
                    },
                    "4_walking": {
                        "duration_seconds": float(wr.walking_duration),
                        "duration_formatted": self._format_duration(wr.walking_duration),
                        "percentage": float(self._pct(wr.walking_duration, wr.total_tracked_time)),
                    },
                    "5_load_carried": {
                        "total_events": int(wr.total_load_events),
                        "instances": [
                            {
                                "timestamp": float(li.timestamp),
                                "object_class": str(li.object_class),
                                "weight_kg": float(li.weight_kg) if li.weight_kg is not None else None,
                            }
                            for li in wr.load_instances
                        ],
                    },
                    "6_repetitive_movement": {
                        "is_repetitive": (
                            bool(wr.repetitive_movement.is_repetitive)
                            if wr.repetitive_movement else False
                        ),
                        "cycles_per_minute": (
                            float(wr.repetitive_movement.cycles_per_minute)
                            if wr.repetitive_movement and wr.repetitive_movement.cycles_per_minute is not None else None
                        ),
                        "frequency_hz": (
                            float(wr.repetitive_movement.frequency_hz)
                            if wr.repetitive_movement and wr.repetitive_movement.frequency_hz is not None else None
                        ),
                        "confidence": (
                            float(wr.repetitive_movement.confidence)
                            if wr.repetitive_movement else 0.0
                        ),
                    },
                    "7_trips": {
                        "count": (
                            int(wr.trip_count_result.trip_count)
                            if wr.trip_count_result else 0
                        ),
                        "total_distance_pixels": (
                            float(wr.trip_count_result.total_distance_pixels)
                            if wr.trip_count_result else 0.0
                        ),
                    },
                    "8_tools_equipment": [
                        {
                            "tool_name": str(t.tool_name),
                            "first_seen": float(t.first_seen),
                            "last_seen": float(t.last_seen),
                            "duration": float(t.duration),
                            "detection_count": int(t.detection_count),
                        }
                        for t in wr.tools_used
                    ],
                    "9_posture": {
                        "dominant_posture": (
                            str(wr.posture_summary.dominant_posture.value)
                            if wr.posture_summary else "unknown"
                        ),
                        "distribution": (
                            wr.posture_summary.posture_distribution
                            if wr.posture_summary else {}
                        ),
                        "avg_trunk_flexion_degrees": (
                            float(wr.posture_summary.avg_trunk_flexion)
                            if wr.posture_summary and wr.posture_summary.avg_trunk_flexion is not None else None
                        ),
                        "max_trunk_flexion_degrees": (
                            float(wr.posture_summary.max_trunk_flexion)
                            if wr.posture_summary and wr.posture_summary.max_trunk_flexion is not None else None
                        ),
                    },
                    "10_continuous_work": {
                        "longest_bout_seconds": float(wr.longest_work_bout),
                        "longest_bout_formatted": self._format_duration(
                            wr.longest_work_bout
                        ),
                        "avg_bout_seconds": float(wr.avg_work_bout),
                        "total_work_bouts": int(len(wr.work_bouts)),
                    },
                    "11_rest": {
                        "total_duration_seconds": float(wr.total_rest_duration),
                        "total_duration_formatted": self._format_duration(
                            wr.total_rest_duration
                        ),
                        "rest_count": int(wr.rest_count),
                        "avg_rest_seconds": float(wr.avg_rest_duration),
                    },
                },
                "ergonomic_score": {
                    "reba_score": int(wr.reba_score) if wr.reba_score is not None else None,
                    "reba_risk_level": str(wr.reba_risk_level) if wr.reba_risk_level is not None else None,
                    "rula_score": int(wr.rula_score) if getattr(wr, 'rula_score', None) is not None else None,
                    "rula_action_level": str(getattr(wr, 'rula_action_level', None)) if getattr(wr, 'rula_action_level', None) is not None else None,
                },
                "drudgery_assessment": {
                    "drudgery_index": float(getattr(wr, 'drudgery_index', 0.0) or 0.0),
                    "drudgery_category": str(getattr(wr, 'drudgery_category', 'N/A')),
                    "estimated_fatigue_level": float(getattr(wr, 'fatigue_level', 0.0) or 0.0),
                    "recommendations": getattr(wr, 'drudgery_recommendations', []),
                },
                "niosh_assessment": {
                    "rwl_kg": float(getattr(wr, 'niosh_rwl_kg', 0.0) or 0.0),
                    "lifting_index": float(getattr(wr, 'niosh_lifting_index', 0.0) or 0.0),
                    "l5s1_compression_n": float(getattr(wr, 'l5s1_compression_n', 0.0) or 0.0),
                    "risk_assessment": str(getattr(wr, 'niosh_risk_assessment', 'N/A')),
                },
                "task_classification": {
                    "primary_task": str(getattr(wr, 'classified_task', 'General Agricultural Work')),
                    "confidence": float(getattr(wr, 'task_confidence', 0.6) or 0.6),
                    "hazard_profile": str(getattr(wr, 'task_hazard_profile', 'General Physical Fatigue')),
                },
                "activity_timeline": [
                    {
                        "activity": str(bout.activity.value),
                        "start_time": float(bout.start_time),
                        "end_time": float(bout.end_time),
                        "duration": float(bout.duration),
                        "is_rest": bool(bout.is_rest),
                    }
                    for bout in wr.activity_bouts
                ],
            }
            report["workers"].append(worker_data)

        # Recursively sanitize entire report dictionary
        clean_report = _to_python_native(report)

        # Write to file if path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(clean_report, f, indent=2, ensure_ascii=False)

        return clean_report

    def generate_csv(
        self,
        worker_reports: List[WorkerReport],
        video_metadata: VideoMetadata,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate a CSV summary (one row per worker).

        Args:
            worker_reports: List of reports for all tracked workers.
            video_metadata: Source video metadata.
            output_path: If provided, write CSV to this file.

        Returns:
            CSV content as a string.
        """
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            "Video File",
            "Worker ID",
            "Total Tracked Time (s)",
            "Sitting Duration (s)",
            "Sitting %",
            "Standing Duration (s)",
            "Standing %",
            "Bending Duration (s)",
            "Bending %",
            "Walking Duration (s)",
            "Walking %",
            "Load Events",
            "Repetitive (cycles/min)",
            "Trip Count",
            "Dominant Posture",
            "Avg Trunk Flexion (°)",
            "Max Trunk Flexion (°)",
            "Longest Work Bout (s)",
            "Avg Work Bout (s)",
            "Total Rest Duration (s)",
            "Rest Count",
            "Avg Rest Duration (s)",
            "Tools Detected",
            "REBA Score",
            "Risk Level",
        ])

        for wr in worker_reports:
            tools_str = ", ".join(t.tool_name for t in wr.tools_used)
            writer.writerow([
                video_metadata.filename,
                wr.worker_id,
                wr.total_tracked_time,
                wr.sitting_duration,
                self._pct(wr.sitting_duration, wr.total_tracked_time),
                wr.standing_duration,
                self._pct(wr.standing_duration, wr.total_tracked_time),
                wr.bending_duration,
                self._pct(wr.bending_duration, wr.total_tracked_time),
                wr.walking_duration,
                self._pct(wr.walking_duration, wr.total_tracked_time),
                wr.total_load_events,
                (wr.repetitive_movement.cycles_per_minute
                 if wr.repetitive_movement and wr.repetitive_movement.is_repetitive
                 else "N/A"),
                (wr.trip_count_result.trip_count if wr.trip_count_result else 0),
                (wr.posture_summary.dominant_posture.value
                 if wr.posture_summary else "unknown"),
                (wr.posture_summary.avg_trunk_flexion
                 if wr.posture_summary else "N/A"),
                (wr.posture_summary.max_trunk_flexion
                 if wr.posture_summary else "N/A"),
                wr.longest_work_bout,
                wr.avg_work_bout,
                wr.total_rest_duration,
                wr.rest_count,
                wr.avg_rest_duration,
                tools_str if tools_str else "None",
                wr.reba_score if wr.reba_score else "N/A",
                wr.reba_risk_level if wr.reba_risk_level else "N/A",
            ])

        csv_content = output.getvalue()

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                f.write(csv_content)

        return csv_content

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format seconds into HH:MM:SS string."""
        if seconds <= 0:
            return "00:00:00"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    @staticmethod
    def _pct(value: float, total: float) -> float:
        """Calculate percentage, safely handling zero total."""
        if total <= 0:
            return 0.0
        return round(value / total * 100, 1)
