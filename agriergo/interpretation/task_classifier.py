"""
Automated Agricultural Task Auto-Classifier.

Infers specific agricultural field tasks (Manual Weeding, Fruit Picking, Hoeing/Tilling,
Crop Transport, Pruning/Shearing) based on posture distributions, joint repetition rates,
and detected tool objects.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


@dataclass
class TaskClassificationResult:
    """Result of agricultural task auto-classification."""
    primary_task: str                  # Detected task name
    confidence: float                   # 0.0 to 1.0 confidence score
    description: str                   # Task description
    ergonomic_hazard_profile: str      # Main ergonomic risk hazard


class TaskClassifier:
    """
    Rule-based & heuristic classifier for agricultural field tasks.
    """

    TASKS = {
        "Manual Weeding / Ground Harvesting": {
            "desc": "Ground-level crop picking or weeding involving stooping/squatting posture.",
            "hazard": "High lower back flexion stress & knee joint compression."
        },
        "Overhead Fruit Harvesting / Picking": {
            "desc": "Tree or elevated crop harvesting involving shoulder extension.",
            "hazard": "Upper limb & neck extension stress."
        },
        "Land Tilling / Hoeing": {
            "desc": "Soil cultivation using manual tools (hoe, spade).",
            "hazard": "Repetitive spinal impact & high physical drudgery."
        },
        "Crop Transport / Load Carrying": {
            "desc": "Manual carrying of harvested crop crates, bags, or buckets.",
            "hazard": "L5/S1 lumbar disc compression & excessive cardiac load."
        },
        "Crop Pruning & Cutting": {
            "desc": "Precision cutting/pruning using shears or sickles.",
            "hazard": "Repetitive hand-wrist fatigue & carpal tunnel strain."
        },
        "General Agricultural Work": {
            "desc": "Mixed field activity including walking, standing, and light handling.",
            "hazard": "General physical fatigue."
        }
    }

    def classify_task(
        self,
        posture_distribution: Dict[str, float],   # posture_name -> percentage
        cycles_per_minute: Optional[float],
        repetitive_joint: Optional[str],
        detected_tools: List[str],
        total_load_events: int,
    ) -> TaskClassificationResult:
        """
        Classify field activity task.
        """
        bending_pct = posture_distribution.get("bending", 0.0)
        squatting_pct = posture_distribution.get("squatting", 0.0)
        standing_pct = posture_distribution.get("standing", 0.0)
        walking_pct = posture_distribution.get("walking", 0.0)
        cpm = cycles_per_minute or 0.0

        # Rule 1: Crop Transport / Load Carrying
        if total_load_events > 0 or "backpack" in detected_tools or "suitcase" in detected_tools:
            if walking_pct > 15.0 or standing_pct > 30.0:
                info = self.TASKS["Crop Transport / Load Carrying"]
                return TaskClassificationResult(
                    primary_task="Crop Transport / Load Carrying",
                    confidence=0.88,
                    description=info["desc"],
                    ergonomic_hazard_profile=info["hazard"],
                )

        # Rule 2: Manual Weeding / Ground Harvesting
        if (bending_pct + squatting_pct) > 40.0:
            info = self.TASKS["Manual Weeding / Ground Harvesting"]
            conf = min(0.95, 0.60 + ((bending_pct + squatting_pct) / 200.0))
            return TaskClassificationResult(
                primary_task="Manual Weeding / Ground Harvesting",
                confidence=round(conf, 2),
                description=info["desc"],
                ergonomic_hazard_profile=info["hazard"],
            )

        # Rule 3: Land Tilling / Hoeing
        if cpm > 20.0 and repetitive_joint in ["trunk", "elbow"]:
            info = self.TASKS["Land Tilling / Hoeing"]
            return TaskClassificationResult(
                primary_task="Land Tilling / Hoeing",
                confidence=0.85,
                description=info["desc"],
                ergonomic_hazard_profile=info["hazard"],
            )

        # Rule 4: Overhead Fruit Harvesting / Picking
        if standing_pct > 50.0 and cpm > 15.0 and repetitive_joint == "shoulder":
            info = self.TASKS["Overhead Fruit Harvesting / Picking"]
            return TaskClassificationResult(
                primary_task="Overhead Fruit Harvesting / Picking",
                confidence=0.82,
                description=info["desc"],
                ergonomic_hazard_profile=info["hazard"],
            )

        # Rule 5: Crop Pruning & Cutting
        if cpm > 25.0 and repetitive_joint == "wrist":
            info = self.TASKS["Crop Pruning & Cutting"]
            return TaskClassificationResult(
                primary_task="Crop Pruning & Cutting",
                confidence=0.84,
                description=info["desc"],
                ergonomic_hazard_profile=info["hazard"],
            )

        # Default: General Agricultural Work
        info = self.TASKS["General Agricultural Work"]
        return TaskClassificationResult(
            primary_task="General Agricultural Work",
            confidence=0.60,
            description=info["desc"],
            ergonomic_hazard_profile=info["hazard"],
        )
