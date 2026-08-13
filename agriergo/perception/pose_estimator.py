"""
Pose Estimator — YOLOv8-Pose keypoint extraction.

Wraps Ultralytics YOLOv8-Pose to extract 17 COCO body keypoints
per detected person in each frame.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import POSE_MODEL, POSE_CONFIDENCE


@dataclass
class PersonKeypoints:
    """Keypoints for a single detected person."""
    person_id: Optional[int]          # Tracking ID (None if not tracked)
    keypoints: np.ndarray             # Shape: (17, 2) — x, y pixel coords
    confidences: np.ndarray           # Shape: (17,) — per-keypoint confidence
    bbox: np.ndarray                  # Shape: (4,) — x1, y1, x2, y2
    bbox_confidence: float            # Detection confidence for the person


class PoseEstimator:
    """
    Multi-person pose estimation using YOLOv8-Pose.

    Extracts 17 COCO keypoints per detected person in a frame.

    COCO Keypoint Layout:
        0: Nose, 1: L-Eye, 2: R-Eye, 3: L-Ear, 4: R-Ear,
        5: L-Shoulder, 6: R-Shoulder, 7: L-Elbow, 8: R-Elbow,
        9: L-Wrist, 10: R-Wrist, 11: L-Hip, 12: R-Hip,
        13: L-Knee, 14: R-Knee, 15: L-Ankle, 16: R-Ankle
    """

    def __init__(self, model_path: str = POSE_MODEL, confidence: float = POSE_CONFIDENCE):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.confidence = confidence

    def estimate(self, frame: np.ndarray) -> List[PersonKeypoints]:
        """
        Run pose estimation on a single frame.

        Args:
            frame: BGR image array (H, W, 3).

        Returns:
            List of PersonKeypoints for each detected person.
        """
        results = self.model(frame, verbose=False, conf=self.confidence)

        persons = []
        for result in results:
            if result.keypoints is None or len(result.keypoints) == 0:
                continue

            keypoints_xy = result.keypoints.xy.cpu().numpy()       # (N, 17, 2)
            keypoints_conf = result.keypoints.conf.cpu().numpy()   # (N, 17)
            boxes = result.boxes.xyxy.cpu().numpy()                # (N, 4)
            box_confs = result.boxes.conf.cpu().numpy()            # (N,)

            for i in range(len(keypoints_xy)):
                persons.append(PersonKeypoints(
                    person_id=None,
                    keypoints=keypoints_xy[i],
                    confidences=keypoints_conf[i],
                    bbox=boxes[i],
                    bbox_confidence=float(box_confs[i]),
                ))

        return persons
