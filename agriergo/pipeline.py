"""
AgriErgo Pipeline — End-to-end video processing orchestration.

Coordinates all three processing layers (Perception → Interpretation → Analytics)
to process a video and produce structured WorkerReports with all 11 parameters.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any, Tuple
from pathlib import Path
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import FRAME_SAMPLE_FPS, RESULTS_DIR

# Perception
from agriergo.perception.video_processor import VideoProcessor, VideoMetadata
from agriergo.perception.tracker import PersonTracker
from agriergo.perception.object_detector import ObjectDetector, DetectedObject
from agriergo.perception.pose_estimator import PersonKeypoints

# Interpretation
from agriergo.interpretation.joint_angles import compute_joint_angles, JointAngles
from agriergo.interpretation.posture_classifier import PostureClassifier, PostureLabel
from agriergo.interpretation.repetition_detector import RepetitionDetector
from agriergo.interpretation.activity_segmenter import (
    ActivitySegmenter, FrameRecord,
)
from agriergo.interpretation.trip_counter import TripCounter

# Analytics
from agriergo.analytics.parameter_aggregator import ParameterAggregator, WorkerReport
from agriergo.analytics.ergonomic_scorer import ErgonomicScorer, REBAScore
from agriergo.analytics.report_generator import ReportGenerator


@dataclass
class PipelineResult:
    """Complete result of pipeline processing."""
    video_metadata: VideoMetadata
    worker_reports: List[WorkerReport]
    processing_time_seconds: float
    frames_processed: int
    workers_detected: int
    json_report: Optional[Dict[str, Any]] = None
    csv_report: Optional[str] = None


class AgriErgoPipeline:
    """
    End-to-end video processing pipeline.

    Workflow:
    1. INGEST: Open video, extract metadata
    2. PROCESS FRAMES: For each sampled frame:
       - Track persons with persistent IDs
       - Detect objects (tools, loads)
       - Compute joint angles
       - Classify posture
       - Accumulate per-worker time-series data
    3. ANALYZE: After all frames:
       - Segment activities into bouts
       - Detect repetitive motions
       - Count trips
       - Aggregate 11 parameters
       - Compute REBA scores
    4. REPORT: Generate JSON/CSV output
    """

    def __init__(self, sample_fps: float = FRAME_SAMPLE_FPS):
        self.sample_fps = sample_fps

        # Initialize components
        self.tracker = PersonTracker()
        self.object_detector = ObjectDetector()
        self.posture_classifier = PostureClassifier()
        self.repetition_detector = RepetitionDetector()
        self.activity_segmenter = ActivitySegmenter()
        self.trip_counter = TripCounter()
        self.aggregator = ParameterAggregator()
        self.scorer = ErgonomicScorer()
        self.report_gen = ReportGenerator()

    def process(
        self,
        video_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> PipelineResult:
        """
        Process a video end-to-end.

        Args:
            video_path: Path to the video file.
            progress_callback: Optional callback(progress_fraction, status_message).

        Returns:
            PipelineResult with all worker reports and metadata.
        """
        start_time = time.time()

        # ════════════════════════════════════════
        # PHASE 1: INGEST
        # ════════════════════════════════════════
        self._report_progress(progress_callback, 0.0, "Opening video...")
        video_proc = VideoProcessor(video_path)
        metadata = video_proc.metadata
        total_frames = video_proc.get_total_sampled_frames(self.sample_fps)

        # ════════════════════════════════════════
        # PHASE 2: PROCESS FRAMES
        # ════════════════════════════════════════
        self._report_progress(progress_callback, 0.05, "Processing frames...")

        # Per-worker accumulators
        worker_frame_records: Dict[int, List[FrameRecord]] = defaultdict(list)
        worker_trajectories: Dict[int, List[Tuple[float, float, float]]] = defaultdict(list)
        worker_tool_detections: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        worker_load_detections: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        worker_trunk_flexions: Dict[int, List[Optional[float]]] = defaultdict(list)
        worker_hip_angles: Dict[int, List[Optional[float]]] = defaultdict(list)
        worker_elbow_angles: Dict[int, List[Optional[float]]] = defaultdict(list)
        worker_reba_scores: Dict[int, List[REBAScore]] = defaultdict(list)

        frames_processed = 0

        for frame_idx, timestamp, frame in video_proc.sample_frames(self.sample_fps):
            # ── Track persons ──
            persons = self.tracker.track(frame)

            # ── Detect objects (run on every Nth frame to save compute) ──
            detected_objects: List[DetectedObject] = []
            if frames_processed % 3 == 0:  # Object detection every 3rd sampled frame
                detected_objects = self.object_detector.detect(frame)

            # ── Process each detected person ──
            for person in persons:
                wid = person.person_id
                if wid is None:
                    continue  # Skip untracked detections

                # Compute joint angles
                angles = compute_joint_angles(
                    person.keypoints, person.confidences
                )

                # Classify posture
                posture_result = self.posture_classifier.classify(
                    worker_id=wid,
                    joint_angles=angles,
                    keypoints=person.keypoints,
                    confidences=person.confidences,
                )

                # Store frame record
                worker_frame_records[wid].append(FrameRecord(
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    posture=posture_result.label,
                    displacement=posture_result.displacement,
                ))

                # Store trajectory (centroid position)
                centroid = TripCounter.extract_centroid(
                    person.keypoints, person.confidences
                )
                if centroid is not None:
                    worker_trajectories[wid].append((timestamp, centroid[0], centroid[1]))

                # Store angle data
                worker_trunk_flexions[wid].append(angles.trunk_flexion)
                worker_hip_angles[wid].append(angles.avg_hip_angle)
                worker_elbow_angles[wid].append(
                    angles.left_elbow_angle or angles.right_elbow_angle
                )

                # Compute per-frame REBA score
                reba = self.scorer.score_frame(angles)
                worker_reba_scores[wid].append(reba)

                # Associate detected objects with this person
                if detected_objects:
                    for obj in detected_objects:
                        # Check if object is near this person
                        ox, oy = obj.center
                        px1, py1, px2, py2 = person.bbox
                        pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
                        diag = np.sqrt((px2 - px1)**2 + (py2 - py1)**2)

                        if np.sqrt((ox - pcx)**2 + (oy - pcy)**2) < diag * 0.6:
                            # Object is near this person
                            worker_tool_detections[wid][obj.class_name].append(timestamp)

                            # Check if it's a load-type object
                            if obj.class_name in {"backpack", "handbag", "suitcase"}:
                                worker_load_detections[wid].append({
                                    "timestamp": timestamp,
                                    "object_class": obj.class_name,
                                })

            frames_processed += 1

            # Progress update
            if total_frames > 0 and frames_processed % 10 == 0:
                pct = 0.05 + (frames_processed / total_frames) * 0.65
                self._report_progress(
                    progress_callback, pct,
                    f"Processed {frames_processed}/{total_frames} frames "
                    f"({len(worker_frame_records)} workers detected)"
                )

        # ════════════════════════════════════════
        # PHASE 3: ANALYZE
        # ════════════════════════════════════════
        self._report_progress(progress_callback, 0.72, "Analyzing activities...")

        worker_reports: List[WorkerReport] = []

        for wid in sorted(worker_frame_records.keys()):
            records = worker_frame_records[wid]
            if len(records) < 5:
                continue  # Skip workers seen in very few frames

            # Segment activities
            bouts = self.activity_segmenter.segment(wid, records)

            # Detect repetitive motions (using elbow angle series)
            elbow_series = np.array([
                a if a is not None else np.nan
                for a in worker_elbow_angles[wid]
            ])
            rep_result = self.repetition_detector.detect_frequency(
                elbow_series, self.sample_fps
            )

            # Count trips
            trajectory = worker_trajectories[wid]
            trip_result = self.trip_counter.count_trips(trajectory)

            # Aggregate parameters
            report = self.aggregator.aggregate(
                worker_id=wid,
                activity_bouts=bouts,
                repetition_result=rep_result,
                trip_result=trip_result,
                tool_detections=dict(worker_tool_detections[wid]),
                load_detections=worker_load_detections[wid],
                trunk_flexions=worker_trunk_flexions[wid],
                hip_angles=worker_hip_angles[wid],
            )

            # Compute overall REBA score
            if worker_reba_scores[wid]:
                overall_reba = self.scorer.score_worker_overall(
                    worker_reba_scores[wid]
                )
                report.reba_score = overall_reba.final_score
                report.reba_risk_level = overall_reba.risk_level

            worker_reports.append(report)

        self._report_progress(progress_callback, 0.90, "Generating reports...")

        # ════════════════════════════════════════
        # PHASE 4: REPORT
        # ════════════════════════════════════════
        # Generate JSON report
        video_name = Path(video_path).stem
        json_path = str(RESULTS_DIR / f"{video_name}_report.json")
        csv_path = str(RESULTS_DIR / f"{video_name}_report.csv")

        json_report = self.report_gen.generate_json(
            worker_reports, metadata, json_path
        )
        csv_report = self.report_gen.generate_csv(
            worker_reports, metadata, csv_path
        )

        elapsed = round(time.time() - start_time, 2)
        self._report_progress(
            progress_callback, 1.0,
            f"Complete! Processed {frames_processed} frames in {elapsed}s. "
            f"Found {len(worker_reports)} workers."
        )

        return PipelineResult(
            video_metadata=metadata,
            worker_reports=worker_reports,
            processing_time_seconds=elapsed,
            frames_processed=frames_processed,
            workers_detected=len(worker_reports),
            json_report=json_report,
            csv_report=csv_report,
        )

    @staticmethod
    def _report_progress(
        callback: Optional[Callable], progress: float, message: str
    ):
        """Report progress if callback is provided."""
        if callback:
            callback(min(1.0, progress), message)
