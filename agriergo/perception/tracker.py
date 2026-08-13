"""
Person Tracker — ByteTrack multi-person tracking with persistent IDs.

Combines YOLOv8-Pose with ByteTrack to assign persistent identity
to each worker across frames so all measurements can be attributed
to individual workers over time.
"""

import numpy as np
from typing import List, Optional, Dict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    POSE_MODEL,
    TRACKER_CONFIG,
    TRACKING_CONFIDENCE,
)
from agriergo.perception.pose_estimator import PersonKeypoints


class PersonTracker:
    """
    Multi-person tracking with persistent IDs using YOLOv8-Pose + ByteTrack.

    Wraps Ultralytics `.track()` to run pose estimation and ByteTrack
    in a single call, producing keypoints with persistent worker IDs.

    Usage:
        tracker = PersonTracker()
        for frame_idx, timestamp, frame in video_processor.sample_frames():
            persons = tracker.track(frame)
            for person in persons:
                print(f"Worker {person.person_id}: {person.keypoints.shape}")
    """

    def __init__(
        self,
        model_path: str = POSE_MODEL,
        tracker_config: str = TRACKER_CONFIG,
        confidence: float = TRACKING_CONFIDENCE,
    ):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.tracker_config = tracker_config
        self.confidence = confidence
        self._active_ids: Dict[int, int] = {}  # Track ID → frame count

    def track(self, frame: np.ndarray) -> List[PersonKeypoints]:
        """
        Run pose estimation + tracking on a frame.

        Must be called sequentially on consecutive frames for
        ByteTrack to maintain ID persistence.

        Args:
            frame: BGR image array (H, W, 3).

        Returns:
            List of PersonKeypoints with persistent person_id set.
        """
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_config,
            verbose=False,
            conf=self.confidence,
        )

        persons = []
        for result in results:
            if result.keypoints is None or len(result.keypoints) == 0:
                continue

            keypoints_xy = result.keypoints.xy.cpu().numpy()
            keypoints_conf = result.keypoints.conf.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            box_confs = result.boxes.conf.cpu().numpy()

            # Get tracking IDs (may be None if tracking fails for a frame)
            track_ids = None
            if result.boxes.id is not None:
                track_ids = result.boxes.id.cpu().numpy().astype(int)

            for i in range(len(keypoints_xy)):
                pid = int(track_ids[i]) if track_ids is not None else None

                # Track how many frames each ID has been seen
                if pid is not None:
                    self._active_ids[pid] = self._active_ids.get(pid, 0) + 1

                persons.append(PersonKeypoints(
                    person_id=pid,
                    keypoints=keypoints_xy[i],
                    confidences=keypoints_conf[i],
                    bbox=boxes[i],
                    bbox_confidence=float(box_confs[i]),
                ))

        return persons

    def reset(self):
        """Reset tracker state (call when starting a new video)."""
        self._active_ids.clear()
        # Reinitialize model to clear ByteTrack internal state
        from ultralytics import YOLO
        self.model = YOLO(self.model.model_name if hasattr(self.model, 'model_name') else POSE_MODEL)

    @property
    def active_worker_ids(self) -> List[int]:
        """Return list of all worker IDs seen so far."""
        return sorted(self._active_ids.keys())

    def get_frame_count(self, worker_id: int) -> int:
        """Return how many frames a specific worker has been detected in."""
        return self._active_ids.get(worker_id, 0)
