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
from agriergo.interpretation.joint_angles import compute_joint_angles


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
    Renders visual overlays and live angle measurements on video frames.
    """

    @staticmethod
    def annotate_frame(
        frame: np.ndarray,
        persons: List[PersonKeypoints],
        posture_labels: dict = None,   # person_id -> PostureLabel
        reba_scores: dict = None,       # person_id -> final_score
    ) -> np.ndarray:
        """
        Draw skeleton overlays, live joint angles, and HUD metrics on a BGR video frame.

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

            # Compute live angles
            angles = compute_joint_angles(person.keypoints, person.confidences)
            trunk_angle = angles.trunk_flexion if angles.trunk_flexion is not None else None
            knee_angle = angles.avg_knee_angle if angles.avg_knee_angle is not None else None

            # Color code based on REBA Risk Level & Trunk Stooping Angle
            if trunk_angle is not None and trunk_angle > 60.0:
                color = (0, 0, 255)      # Red (Severe plantation stoop / High Risk >60°)
            elif (trunk_angle is not None and trunk_angle > 20.0) or score > 3:
                color = (0, 215, 255)    # Yellow/Gold (Moderate bending 20-60°)
            else:
                color = (0, 220, 0)      # Green (Safe upright 0-20°)

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

            # 3. Draw Live Joint Angle Callouts on Skeleton
            # Mid-spine position for Trunk Bending Angle
            if conf[5] > 0.3 and conf[6] > 0.3 and conf[11] > 0.3 and conf[12] > 0.3 and trunk_angle is not None:
                mid_sh = ((kp[5][0] + kp[6][0]) / 2, (kp[5][1] + kp[6][1]) / 2)
                mid_hip = ((kp[11][0] + kp[12][0]) / 2, (kp[11][1] + kp[12][1]) / 2)
                spine_center = (int((mid_sh[0] + mid_hip[0]) / 2), int((mid_sh[1] + mid_hip[1]) / 2))

                angle_tag = f"Trunk: {round(trunk_angle, 1)} deg"
                (atw, ath), _ = cv2.getTextSize(angle_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(
                    annotated,
                    (spine_center[0] + 8, spine_center[1] - ath - 4),
                    (spine_center[0] + atw + 14, spine_center[1] + 4),
                    (20, 20, 20),
                    -1
                )
                cv2.rectangle(
                    annotated,
                    (spine_center[0] + 8, spine_center[1] - ath - 4),
                    (spine_center[0] + atw + 14, spine_center[1] + 4),
                    color,
                    1
                )
                cv2.putText(
                    annotated,
                    angle_tag,
                    (spine_center[0] + 11, spine_center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1
                )

            # Knee position callout
            if (conf[13] > 0.3 or conf[14] > 0.3) and knee_angle is not None:
                knee_pt = kp[13] if conf[13] > 0.3 else kp[14]
                kpt = (int(knee_pt[0]), int(knee_pt[1]))
                knee_tag = f"Knee: {round(knee_angle, 0)} deg"
                cv2.putText(
                    annotated,
                    knee_tag,
                    (kpt[0] + 10, kpt[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    (240, 240, 240),
                    1
                )

            # Shoulder Elevation angle callout (Arm Postural Study)
            sh_ang = angles.avg_shoulder_angle or angles.left_shoulder_angle or angles.right_shoulder_angle
            if (conf[5] > 0.3 or conf[6] > 0.3) and sh_ang is not None:
                sh_pt = kp[5] if conf[5] > 0.3 else kp[6]
                spt = (int(sh_pt[0]), int(sh_pt[1]))
                sh_color = (0, 0, 255) if sh_ang >= 45.0 else (0, 220, 0)
                if sh_ang >= 90.0:
                    sh_tag = f"Shoulder: {round(sh_ang, 0)} deg [OVERHEAD]"
                    sh_color = (0, 0, 255)
                else:
                    sh_tag = f"Shoulder: {round(sh_ang, 0)} deg"
                cv2.putText(
                    annotated,
                    sh_tag,
                    (spt[0] - 80, spt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    sh_color,
                    1
                )

            # Elbow Flexion angle callout
            el_ang = angles.avg_elbow_angle or angles.left_elbow_angle or angles.right_elbow_angle
            if (conf[7] > 0.3 or conf[8] > 0.3) and el_ang is not None:
                el_pt = kp[7] if conf[7] > 0.3 else kp[8]
                ept = (int(el_pt[0]), int(el_pt[1]))
                el_tag = f"Elbow: {round(el_ang, 0)} deg"
                cv2.putText(
                    annotated,
                    el_tag,
                    (ept[0] + 10, ept[1] + 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.36,
                    (200, 230, 255),
                    1
                )

            # 4. Draw Bounding Box & HUD Label Badge
            if person.bbox is not None and len(person.bbox) == 4:
                x1, y1, x2, y2 = [int(v) for v in person.bbox]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                # Badge label text
                wid_str = f"Worker #{wid}" if wid is not None else "Worker"
                posture_str = posture.value.title() if hasattr(posture, 'value') else str(posture).title()
                trunk_str = f"{round(trunk_angle, 0)} deg" if trunk_angle is not None else "N/A"
                badge_text = f"{wid_str} | {posture_str} (Trunk: {trunk_str}) | REBA:{score}"

                # Text background rectangle
                (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(annotated, (x1, max(0, y1 - 25)), (x1 + tw + 10, y1), color, -1)
                cv2.putText(
                    annotated,
                    badge_text,
                    (x1 + 5, max(15, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (255, 255, 255),
                    2,
                )

        return annotated
