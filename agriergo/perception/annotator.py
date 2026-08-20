"""
Pose Skeleton & HUD Video Annotator.

Draws color-coded skeleton overlays, bounding boxes, active joint angles,
worker IDs, and risk score badges (Green / Yellow / Red) onto video frames.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from agriergo.perception.pose_estimator import PersonKeypoints
from agriergo.interpretation.posture_classifier import PostureLabel


# COCO Skeleton Bone Connections (pairs of keypoint indices)
SKELETON_BONES = [
    # Spine & Head
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 11), (6, 12), (11, 12),
    # Left Arm
    (5, 7), (7, 9),
    # Right Arm
    (6, 8), (8, 10),
    # Left Leg
    (11, 13), (13, 15),
    # Right Leg
    (12, 14), (14, 16),
]


class VideoAnnotator:
    """
    Renders visual overlays on video frames.
    """

    @staticmethod
    def annotate_frame(
        frame: np.ndarray,
        persons: List[PersonKeypoints],
        posture_labels: dict = None,   # person_id -> PostureLabel
        reba_scores: dict = None,       # person_id -> final_score
    ) -> np.ndarray:
        """
        Draw skeleton overlays and HUD metrics on a BGR video frame.

        Args:
            frame: Input BGR image array.
            persons: List of PersonKeypoints detected in frame.
            posture_labels: Dict of worker_id -> PostureLabel.
            reba_scores: Dict of worker_id -> REBA score.

        Returns:
            Annotated BGR frame array.
        """
        annotated = frame.copy()
        posture_labels = posture_labels or {}
        reba_scores = reba_scores or {}

        for person in persons:
            wid = person.person_id
            score = reba_scores.get(wid, 1) if wid is not None else 1
            posture = posture_labels.get(wid, PostureLabel.UNKNOWN)

            # Color code based on REBA Risk Level
            if score <= 3:
                color = (0, 220, 0)      # Green (Low risk)
            elif score <= 7:
                color = (0, 215, 255)    # Yellow/Gold (Medium risk)
            else:
                color = (0, 0, 255)      # Red (High risk)

            # 1. Draw Skeleton Bones
            kp = person.keypoints
            conf = person.confidences

            for p1_idx, p2_idx in SKELETON_BONES:
                if conf[p1_idx] > 0.3 and conf[p2_idx] > 0.3:
                    pt1 = (int(kp[p1_idx][0]), int(kp[p1_idx][1]))
                    pt2 = (int(kp[p2_idx][0]), int(kp[p2_idx][1]))
                    cv2.line(annotated, pt1, pt2, color, 3)

            # 2. Draw Keypoint Joints
            for i in range(17):
                if conf[i] > 0.3:
                    pt = (int(kp[i][0]), int(kp[i][1]))
                    cv2.circle(annotated, pt, 5, (255, 255, 255), -1)
                    cv2.circle(annotated, pt, 5, color, 2)

            # 3. Draw Bounding Box & HUD Label Badge
            if person.bbox is not None and len(person.bbox) == 4:
                x1, y1, x2, y2 = [int(v) for v in person.bbox]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                # Badge label text
                wid_str = f"Worker #{wid}" if wid is not None else "Worker"
                posture_str = posture.value.title() if hasattr(posture, 'value') else str(posture).title()
                badge_text = f"{wid_str} | {posture_str} | REBA:{score}"

                # Text background rectangle
                (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x1, max(0, y1 - 25)), (x1 + tw + 10, y1), color, -1)
                cv2.putText(
                    annotated,
                    badge_text,
                    (x1 + 5, max(15, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )

        return annotated
