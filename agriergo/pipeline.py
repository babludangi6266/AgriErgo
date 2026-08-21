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
from config.settings import FRAME_SAMPLE_FPS, RESULTS_DIR, OBJECT_DETECTION_STRIDE

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
from agriergo.analytics.rula_scorer import RULAScorer, RULAScore
from agriergo.analytics.drudgery_index import DrudgeryCalculator, DrudgeryResult
from agriergo.analytics.niosh_calculator import NIOSHCalculator, NIOSHResult
from agriergo.interpretation.task_classifier import TaskClassifier, TaskClassificationResult
from agriergo.analytics.report_generator import ReportGenerator
from agriergo.analytics.pdf_generator import PDFReportGenerator
from agriergo.perception.annotator import VideoAnnotator
from agriergo.perception.tracklet_stitcher import TrackletStitcher


@dataclass
class PipelineResult:
    """Complete result of pipeline processing."""
    video_metadata: VideoMetadata
    worker_reports: List[WorkerReport]
    processing_time_seconds: float
    frames_processed: int
    workers_detected: int
    peak_concurrent_workers: int = 0
    json_report: Optional[Dict[str, Any]] = None
    csv_report: Optional[str] = None
    pdf_report: Optional[bytes] = None
    annotated_video_bytes: Optional[bytes] = None


class AgriErgoPipeline:
    """
    End-to-end video processing pipeline.
    """

    def __init__(self, sample_fps: float = FRAME_SAMPLE_FPS):
        self.sample_fps = sample_fps

        # Initialize perception components
        self.tracker = PersonTracker()
        self.object_detector = ObjectDetector()
        self.annotator = VideoAnnotator()

        # Initialize interpretation components
        self.posture_classifier = PostureClassifier()
        self.repetition_detector = RepetitionDetector()
        self.activity_segmenter = ActivitySegmenter()
        self.trip_counter = TripCounter()
        self.task_classifier = TaskClassifier()

        # Initialize analytics components
        self.aggregator = ParameterAggregator()
        self.scorer = ErgonomicScorer()
        self.rula_scorer = RULAScorer()
        self.drudgery_calculator = DrudgeryCalculator()
        self.niosh_calculator = NIOSHCalculator()

        # Initialize reporting components
        self.report_gen = ReportGenerator()
        self.pdf_gen = PDFReportGenerator()

    def process(
        self,
        video_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        speed_mode: str = "Balanced Fast",
        **kwargs,
    ) -> PipelineResult:
        """
        Process a video end-to-end with high-speed CPU optimization.

        Args:
            video_path: Path to the video file.
            progress_callback: Optional callback(progress_fraction, status_message).
            speed_mode: Speed profile ("Lightning Fast", "Balanced Fast", "High Precision Research").

        Returns:
            PipelineResult with all worker reports and metadata.
        """
        start_time = time.time()

        # Multi-thread CPU optimization for PyTorch
        try:
            import torch
            import os
            torch.set_num_threads(max(1, os.cpu_count() or 4))
        except Exception:
            pass

        # ════════════════════════════════════════
        # PHASE 1: INGEST
        # ════════════════════════════════════════
        self._report_progress(progress_callback, 0.0, "Opening video...")
        video_proc = VideoProcessor(video_path)
        metadata = video_proc.metadata
        from config.settings import get_adaptive_fps
        effective_fps = get_adaptive_fps(metadata.duration_seconds, speed_mode=speed_mode)
        total_frames = video_proc.get_total_sampled_frames(effective_fps)

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
        worker_knee_angles: Dict[int, List[Optional[float]]] = defaultdict(list)
        worker_timestamps: Dict[int, List[float]] = defaultdict(list)
        worker_elbow_angles: Dict[int, List[Optional[float]]] = defaultdict(list)
        worker_wrist_angles: Dict[int, List[Optional[float]]] = defaultdict(list)
        worker_shoulder_angles: Dict[int, List[Optional[float]]] = defaultdict(list)
        worker_reba_scores: Dict[int, List[REBAScore]] = defaultdict(list)

        frames_processed = 0

        for frame_idx, timestamp, frame in video_proc.sample_frames(speed_mode=speed_mode):
            # ── Track persons ──
            persons = self.tracker.track(frame)

            # ── Detect objects (run on every Nth frame to save compute) ──
            detected_objects: List[DetectedObject] = []
            if frames_processed % OBJECT_DETECTION_STRIDE == 0:  # Object detection every 5th sampled frame
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

                # Store multi-joint angle series
                worker_trunk_flexions[wid].append(angles.trunk_flexion)
                worker_hip_angles[wid].append(angles.avg_hip_angle)
                worker_knee_angles[wid].append(angles.avg_knee_angle)
                worker_timestamps[wid].append(timestamp)
                worker_elbow_angles[wid].append(
                    angles.left_elbow_angle or angles.right_elbow_angle
                )
                worker_wrist_angles[wid].append(
                    angles.left_wrist_angle or angles.right_wrist_angle
                )
                worker_shoulder_angles[wid].append(
                    angles.left_shoulder_angle or angles.right_shoulder_angle
                )

                # Check if person carries load to pass load_force to REBA
                has_load = any(
                    det["timestamp"] == timestamp for det in worker_load_detections[wid]
                )
                load_force = 2 if has_load else 0

                # Compute per-frame REBA score with load adjustment
                reba = self.scorer.score_frame(angles, load_force=load_force)
                worker_reba_scores[wid].append(reba)

                # Associate detected objects with this person
                if detected_objects:
                    for obj in detected_objects:
                        ox, oy = obj.center
                        px1, py1, px2, py2 = person.bbox
                        pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
                        diag = np.sqrt((px2 - px1)**2 + (py2 - py1)**2)

                        if np.sqrt((ox - pcx)**2 + (oy - pcy)**2) < diag * 0.6:
                            worker_tool_detections[wid][obj.class_name].append(timestamp)

                            if obj.class_name in {"backpack", "handbag", "suitcase"}:
                                worker_load_detections[wid].append({
                                    "timestamp": timestamp,
                                    "object_class": obj.class_name,
                                })

            frames_processed += 1

            if total_frames > 0 and frames_processed % 10 == 0:
                pct = 0.05 + (frames_processed / total_frames) * 0.65
                self._report_progress(
                    progress_callback, pct,
                    f"Processed {frames_processed}/{total_frames} frames "
                    f"({len(worker_frame_records)} workers detected)"
                )

        # ════════════════════════════════════════
        # PHASE 3: ANALYZE & STITCH TRACKLETS
        # ════════════════════════════════════════
        self._report_progress(progress_callback, 0.70, "Consolidating worker trajectories...")

        # Spatial-temporal tracklet stitching to merge fragmented IDs
        stitcher = TrackletStitcher()
        raw_to_unified, peak_concurrency = stitcher.stitch_tracklets(worker_trajectories)

        # Merge accumulators under unified worker IDs
        unified_frame_records: Dict[int, List[FrameRecord]] = defaultdict(list)
        unified_trajectories: Dict[int, List[Tuple[float, float, float]]] = defaultdict(list)
        unified_tool_detections: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        unified_load_detections: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        unified_trunk_flexions: Dict[int, List[Optional[float]]] = defaultdict(list)
        unified_hip_angles: Dict[int, List[Optional[float]]] = defaultdict(list)
        unified_knee_angles: Dict[int, List[Optional[float]]] = defaultdict(list)
        unified_timestamps: Dict[int, List[float]] = defaultdict(list)
        unified_elbow_angles: Dict[int, List[Optional[float]]] = defaultdict(list)
        unified_wrist_angles: Dict[int, List[Optional[float]]] = defaultdict(list)
        unified_shoulder_angles: Dict[int, List[Optional[float]]] = defaultdict(list)
        unified_reba_scores: Dict[int, List[REBAScore]] = defaultdict(list)

        for raw_id in worker_frame_records.keys():
            unified_id = raw_to_unified.get(raw_id, raw_id)
            unified_frame_records[unified_id].extend(worker_frame_records[raw_id])
            unified_trajectories[unified_id].extend(worker_trajectories[raw_id])
            for tool_name, ts_list in worker_tool_detections[raw_id].items():
                unified_tool_detections[unified_id][tool_name].extend(ts_list)
            unified_load_detections[unified_id].extend(worker_load_detections[raw_id])
            unified_trunk_flexions[unified_id].extend(worker_trunk_flexions[raw_id])
            unified_hip_angles[unified_id].extend(worker_hip_angles[raw_id])
            unified_knee_angles[unified_id].extend(worker_knee_angles[raw_id])
            unified_timestamps[unified_id].extend(worker_timestamps[raw_id])
            unified_elbow_angles[unified_id].extend(worker_elbow_angles[raw_id])
            unified_wrist_angles[unified_id].extend(worker_wrist_angles[raw_id])
            unified_shoulder_angles[unified_id].extend(worker_shoulder_angles[raw_id])
            unified_reba_scores[unified_id].extend(worker_reba_scores[raw_id])

        # Sort merged records chronologically for each unified worker
        for uid in unified_frame_records:
            unified_frame_records[uid].sort(key=lambda r: r.timestamp)
            unified_trajectories[uid].sort(key=lambda t: t[0])
            unified_load_detections[uid].sort(key=lambda d: d["timestamp"])

        self._report_progress(progress_callback, 0.75, "Analyzing activities & ergonomic scores...")

        worker_reports: List[WorkerReport] = []

        for wid in sorted(unified_frame_records.keys()):
            records = unified_frame_records[wid]
            if len(records) < 3:
                continue  # Skip fleeting noise detections

            # Segment activities
            bouts = self.activity_segmenter.segment(wid, records)

            # Detect multi-joint repetitive motions (elbow, wrist, shoulder, trunk)
            multi_joint_series = {
                "elbow": np.array([a if a is not None else np.nan for a in unified_elbow_angles[wid]]),
                "wrist": np.array([a if a is not None else np.nan for a in unified_wrist_angles[wid]]),
                "shoulder": np.array([a if a is not None else np.nan for a in unified_shoulder_angles[wid]]),
                "trunk": np.array([a if a is not None else np.nan for a in unified_trunk_flexions[wid]]),
            }
            rep_result = self.repetition_detector.detect_multi_joint(
                multi_joint_series, self.sample_fps
            )

            # Count trips
            trajectory = unified_trajectories[wid]
            trip_result = self.trip_counter.count_trips(trajectory)

            # Aggregate parameters
            report = self.aggregator.aggregate(
                worker_id=wid,
                activity_bouts=bouts,
                repetition_result=rep_result,
                trip_result=trip_result,
                tool_detections=dict(unified_tool_detections[wid]),
                load_detections=unified_load_detections[wid],
                trunk_flexions=unified_trunk_flexions[wid],
                hip_angles=unified_hip_angles[wid],
                knee_angles=unified_knee_angles[wid],
                timestamps=unified_timestamps[wid],
            )

            # Compute overall REBA score
            if unified_reba_scores[wid]:
                overall_reba = self.scorer.score_worker_overall(
                    unified_reba_scores[wid]
                )
                report.reba_score = overall_reba.final_score
                report.reba_risk_level = overall_reba.risk_level

            # Compute RULA score
            if hasattr(self, 'rula_scorer') and unified_reba_scores[wid]:
                # Approximate RULA from frame joint angles
                rula_scores = [
                    self.rula_scorer.score_frame(
                        JointAngles(trunk_flexion=tf, avg_hip_angle=ha),
                        is_repetitive=rep_result.is_repetitive if rep_result else False
                    )
                    for tf, ha in zip(unified_trunk_flexions[wid], unified_hip_angles[wid])
                ]
                overall_rula = self.rula_scorer.score_worker_overall(rula_scores)
                report.rula_score = overall_rula.final_score
                report.rula_action_level = overall_rula.action_level

            # Compute Agricultural Drudgery Index (ADI)
            if hasattr(self, 'drudgery_calculator'):
                dist = report.posture_summary.posture_distribution if report.posture_summary else {}
                drudgery_res = self.drudgery_calculator.calculate(
                    reba_score=float(report.reba_score or 1),
                    rula_score=float(getattr(report, 'rula_score', 1) or 1),
                    posture_distribution=dist,
                    cycles_per_minute=rep_result.cycles_per_minute if rep_result else 0.0,
                    total_tracked_seconds=report.total_tracked_time,
                    longest_work_bout_seconds=report.longest_work_bout,
                    total_rest_seconds=report.total_rest_duration,
                    load_events_count=report.total_load_events,
                )
                report.drudgery_index = drudgery_res.drudgery_index
                report.drudgery_category = drudgery_res.drudgery_category
                report.drudgery_recommendations = drudgery_res.recommendations
                report.fatigue_level = drudgery_res.estimated_fatigue_level
                report.minute_fatigue_series = drudgery_res.minute_fatigue_series

            # Compute NIOSH & Lumbar Compression Force
            if hasattr(self, 'niosh_calculator'):
                avg_flexion = report.posture_summary.avg_trunk_flexion if report.posture_summary else 0.0
                actual_kg = 5.0 if report.total_load_events > 0 else 0.5
                niosh_res = self.niosh_calculator.calculate(
                    actual_weight_kg=actual_kg,
                    trunk_flexion_degrees=float(avg_flexion or 0.0),
                )
                report.niosh_rwl_kg = niosh_res.recommended_weight_limit_kg
                report.niosh_lifting_index = niosh_res.lifting_index
                report.l5s1_compression_n = niosh_res.l5s1_compression_force_n
                report.niosh_risk_assessment = niosh_res.risk_assessment

            # Agricultural Task Auto-Classification
            if hasattr(self, 'task_classifier'):
                dist = report.posture_summary.posture_distribution if report.posture_summary else {}
                task_res = self.task_classifier.classify_task(
                    posture_distribution=dist,
                    cycles_per_minute=rep_result.cycles_per_minute if rep_result else 0.0,
                    repetitive_joint=rep_result.primary_joint if rep_result else None,
                    detected_tools=[t.tool_name for t in report.tools_used],
                    total_load_events=report.total_load_events,
                )
                report.classified_task = task_res.primary_task
                report.task_confidence = task_res.confidence
                report.task_hazard_profile = task_res.ergonomic_hazard_profile

            # Evaluate ISO 11226 Shift Exposure Limits
            iso_viol, iso_msg = self.scorer.evaluate_shift_iso11226(
                bending_duration_seconds=report.bending_duration,
                total_tracked_seconds=report.total_tracked_time,
            )
            report.iso_11226_violated = iso_viol
            report.iso_11226_message = iso_msg

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
        pdf_path = str(RESULTS_DIR / f"{video_name}_report.pdf")
        pdf_report = self.pdf_gen.generate_pdf(
            worker_reports, metadata, pdf_path
        )

        elapsed = round(time.time() - start_time, 2)
        self._report_progress(
            progress_callback, 1.0,
            f"Complete! Processed {frames_processed} frames in {elapsed}s. "
            f"Found {len(worker_reports)} physical workers (Peak concurrent: {peak_concurrency})."
        )

        return PipelineResult(
            video_metadata=metadata,
            worker_reports=worker_reports,
            processing_time_seconds=elapsed,
            frames_processed=frames_processed,
            workers_detected=len(worker_reports),
            peak_concurrent_workers=peak_concurrency,
            json_report=json_report,
            csv_report=csv_report,
            pdf_report=pdf_report,
        )

    @staticmethod
    def _report_progress(
        callback: Optional[Callable], progress: float, message: str
    ):
        """Report progress if callback is provided."""
        if callback:
            callback(min(1.0, progress), message)
