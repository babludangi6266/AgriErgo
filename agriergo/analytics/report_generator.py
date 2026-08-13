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


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to convert numpy types (bool_, int64, float64, ndarray) to Python types."""
    def default(self, obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


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
                "duration_seconds": video_metadata.duration_seconds,
                "resolution": f"{video_metadata.width}x{video_metadata.height}",
                "fps": video_metadata.fps,
                "total_frames": video_metadata.total_frames,
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
                "worker_id": wr.worker_id,
                "total_tracked_time": wr.total_tracked_time,
                "total_tracked_time_formatted": self._format_duration(
                    wr.total_tracked_time
                ),
                "parameters": {
                    "1_sitting": {
                        "duration_seconds": wr.sitting_duration,
                        "duration_formatted": self._format_duration(wr.sitting_duration),
                        "percentage": self._pct(wr.sitting_duration, wr.total_tracked_time),
                    },
                    "2_standing": {
                        "duration_seconds": wr.standing_duration,
                        "duration_formatted": self._format_duration(wr.standing_duration),
                        "percentage": self._pct(wr.standing_duration, wr.total_tracked_time),
                    },
                    "3_bending": {
                        "duration_seconds": wr.bending_duration,
                        "duration_formatted": self._format_duration(wr.bending_duration),
                        "percentage": self._pct(wr.bending_duration, wr.total_tracked_time),
                    },
                    "4_walking": {
                        "duration_seconds": wr.walking_duration,
                        "duration_formatted": self._format_duration(wr.walking_duration),
                        "percentage": self._pct(wr.walking_duration, wr.total_tracked_time),
                    },
                    "5_load_carried": {
                        "total_events": wr.total_load_events,
                        "instances": [
                            {
                                "timestamp": li.timestamp,
                                "object_class": li.object_class,
                                "weight_kg": li.weight_kg,
                            }
                            for li in wr.load_instances
                        ],
                    },
                    "6_repetitive_movement": {
                        "is_repetitive": (
                            wr.repetitive_movement.is_repetitive
                            if wr.repetitive_movement else False
                        ),
                        "cycles_per_minute": (
                            wr.repetitive_movement.cycles_per_minute
                            if wr.repetitive_movement else None
                        ),
                        "frequency_hz": (
                            wr.repetitive_movement.frequency_hz
                            if wr.repetitive_movement else None
                        ),
                        "confidence": (
                            wr.repetitive_movement.confidence
                            if wr.repetitive_movement else 0.0
                        ),
                    },
                    "7_trips": {
                        "count": (
                            wr.trip_count_result.trip_count
                            if wr.trip_count_result else 0
                        ),
                        "total_distance_pixels": (
                            wr.trip_count_result.total_distance_pixels
                            if wr.trip_count_result else 0
                        ),
                    },
                    "8_tools_equipment": [
                        {
                            "tool_name": t.tool_name,
                            "first_seen": t.first_seen,
                            "last_seen": t.last_seen,
                            "duration": t.duration,
                            "detection_count": t.detection_count,
                        }
                        for t in wr.tools_used
                    ],
                    "9_posture": {
                        "dominant_posture": (
                            wr.posture_summary.dominant_posture.value
                            if wr.posture_summary else "unknown"
                        ),
                        "distribution": (
                            wr.posture_summary.posture_distribution
                            if wr.posture_summary else {}
                        ),
                        "avg_trunk_flexion_degrees": (
                            wr.posture_summary.avg_trunk_flexion
                            if wr.posture_summary else None
                        ),
                        "max_trunk_flexion_degrees": (
                            wr.posture_summary.max_trunk_flexion
                            if wr.posture_summary else None
                        ),
                    },
                    "10_continuous_work": {
                        "longest_bout_seconds": wr.longest_work_bout,
                        "longest_bout_formatted": self._format_duration(
                            wr.longest_work_bout
                        ),
                        "avg_bout_seconds": wr.avg_work_bout,
                        "total_work_bouts": len(wr.work_bouts),
                    },
                    "11_rest": {
                        "total_duration_seconds": wr.total_rest_duration,
                        "total_duration_formatted": self._format_duration(
                            wr.total_rest_duration
                        ),
                        "rest_count": wr.rest_count,
                        "avg_rest_seconds": wr.avg_rest_duration,
                    },
                },
                "ergonomic_score": {
                    "reba_score": wr.reba_score,
                    "risk_level": wr.reba_risk_level,
                },
                "activity_timeline": [
                    {
                        "activity": bout.activity.value,
                        "start_time": bout.start_time,
                        "end_time": bout.end_time,
                        "duration": bout.duration,
                        "is_rest": bout.is_rest,
                    }
                    for bout in wr.activity_bouts
                ],
            }
            report["workers"].append(worker_data)

        # Write to file if path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

        # Convert report to JSON-serializable python dict using NumpyEncoder
        return json.loads(json.dumps(report, cls=NumpyEncoder))

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
