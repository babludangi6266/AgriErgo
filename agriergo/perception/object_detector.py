"""
Object Detector — YOLOv8 tool/load/equipment detection.

Uses COCO-pretrained YOLOv8 to detect objects relevant to agricultural
work (tools, bags, vehicles, etc.) as a placeholder until fine-tuned
on real farm equipment images in Phase 2.
"""

import numpy as np
from dataclasses import dataclass
from typing import List
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    DETECTION_MODEL,
    DETECTION_CONFIDENCE,
    FARM_RELEVANT_COCO_CLASSES,
    FAST_YOLO_IMGSZ,
)


@dataclass
class DetectedObject:
    """A single detected object."""
    class_id: int
    class_name: str
    confidence: float
    bbox: np.ndarray           # x1, y1, x2, y2
    center: tuple              # (cx, cy)


class ObjectDetector:
    """
    Object detection for tools, equipment, and carried loads.

    Uses COCO-pretrained YOLOv8 and filters to farm-relevant classes.
    """

    def __init__(
        self,
        model_path: str = DETECTION_MODEL,
        confidence: float = DETECTION_CONFIDENCE,
    ):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.confidence = confidence
        # Get the COCO class names from the model
        self._model_names = self.model.names  # {id: name}

    def detect(self, frame: np.ndarray) -> List[DetectedObject]:
        """
        Detect farm-relevant objects in a frame.

        Args:
            frame: BGR image array (H, W, 3).

        Returns:
            List of DetectedObject for farm-relevant classes.
        """
        results = self.model(frame, verbose=False, conf=self.confidence, imgsz=FAST_YOLO_IMGSZ)

        objects = []
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            for i in range(len(boxes)):
                cid = int(class_ids[i])
                # Filter to farm-relevant classes only
                if cid not in FARM_RELEVANT_COCO_CLASSES:
                    continue
                # Skip "person" class — handled by pose estimator
                if cid == 0:
                    continue

                x1, y1, x2, y2 = boxes[i]
                objects.append(DetectedObject(
                    class_id=cid,
                    class_name=FARM_RELEVANT_COCO_CLASSES.get(
                        cid, self._model_names.get(cid, f"class_{cid}")
                    ),
                    confidence=float(confs[i]),
                    bbox=boxes[i],
                    center=(float((x1 + x2) / 2), float((y1 + y2) / 2)),
                ))

        return objects

    def detect_near_person(
        self,
        frame: np.ndarray,
        person_bbox: np.ndarray,
        proximity_ratio: float = 0.5,
    ) -> List[DetectedObject]:
        """
        Detect objects near a specific person (for load/tool association).

        Args:
            frame: BGR image.
            person_bbox: Person bounding box (x1, y1, x2, y2).
            proximity_ratio: How close an object must be relative to
                             person bbox diagonal to be considered "near".

        Returns:
            Objects detected within proximity of the person.
        """
        all_objects = self.detect(frame)
        if not all_objects:
            return []

        # Compute person center and diagonal
        px1, py1, px2, py2 = person_bbox
        pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
        diag = np.sqrt((px2 - px1) ** 2 + (py2 - py1) ** 2)
        max_dist = diag * proximity_ratio

        near_objects = []
        for obj in all_objects:
            ocx, ocy = obj.center
            dist = np.sqrt((ocx - pcx) ** 2 + (ocy - pcy) ** 2)
            if dist <= max_dist:
                near_objects.append(obj)

        return near_objects
