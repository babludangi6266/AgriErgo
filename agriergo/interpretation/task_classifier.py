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
        "Land & Bed Preparation (Hoeing / Ridging)": {
            "desc": "Manual land tilling, bed preparation, and hoeing with Kodal/Mattock.",
            "hazard": "Repetitive spinal impact, high lumbar shear, and high physical drudgery.",
            "kinovea_trunk_ref": "45°-65° (Hoeing) / 75°-90° (Clearing)",
            "kinovea_elbow_ref": "60°-120° (Dynamic flexion-extension)",
            "freq_ref": "25-35 strokes/min"
        },
        "Seed Rhizome Preparation & Treatment": {
            "desc": "Sorting, treating, and handling planting material rhizomes in tubs/basins.",
            "hazard": "Static forward flexion while seated/squatting and wrist twisting.",
            "kinovea_trunk_ref": "25°-50° (Forward lean) / 65°-80° (Reaching into tubs)",
            "kinovea_elbow_ref": "70°-110° (Elbow flexion with wrist twisting)",
            "freq_ref": "30-45 movements/min"
        },
        "Sowing & Rhizome Planting": {
            "desc": "Placing treated seed rhizomes into soil beds and covering with soil.",
            "hazard": "Sustained deep squatting/kneeling and repetitive knee/back strain.",
            "kinovea_trunk_ref": "35°-60° (Forward lean squatting/kneeling)",
            "kinovea_elbow_ref": "60°-100° (Arm extension & placement)",
            "freq_ref": "15-25 seed placements/min"
        },
        "Digging & Rhizome Harvesting": {
            "desc": "Harvesting turmeric clumps using hand trowel (Khurpi) and collecting in basins.",
            "hazard": "High lower back flexion stress, wrist deviation, and knee compression.",
            "kinovea_trunk_ref": "35°-60° (Digging) / 60°-85° (Reaching)",
            "kinovea_elbow_ref": "70°-110° (Elbow flexion) / 15°-30° (Wrist deviation)",
            "freq_ref": "15-25 actions/min"
        },
        "Cleaning & Rhizome Separation": {
            "desc": "Trimming roots and detaching mother/finger rhizomes manually.",
            "hazard": "Rapid finger manipulation strain, carpal tunnel fatigue, and squatting strain.",
            "kinovea_trunk_ref": "35°-55° (Forward trunk flexion)",
            "kinovea_elbow_ref": "65°-105° (Elbow flexion & finger detachment)",
            "freq_ref": "40-55 movements/min"
        },
        "Boiling / Parboiling": {
            "desc": "Stirring and turning rhizomes in boiling vessels using long wooden pole.",
            "hazard": "Thermal radiation exposure, continuous shoulder elevation, and static standing.",
            "kinovea_trunk_ref": "10°-25° (Slight forward flexion/standing)",
            "kinovea_elbow_ref": "40°-90° (Circular pushing/stirring motion)",
            "freq_ref": "18-28 stirring strokes/min"
        },
        "Sorting & Grading": {
            "desc": "Grading rhizomes on tarpaulin sheets using winnowing trays (Kula) / crates.",
            "hazard": "Prolonged sitting/squatting with repetitive reaching and neck flexion.",
            "kinovea_trunk_ref": "40°-70° (Forward bending while seated/squatting)",
            "kinovea_elbow_ref": "60°-100° (Repetitive reaching & picking)",
            "freq_ref": "35-50 sorting movements/min"
        },
        "Polishing (Mechanical / Drum)": {
            "desc": "Operating rotary polishing drum, loading rhizomes, and sweeping dust.",
            "hazard": "Deep forward bending, dust inhalation, and repetitive manual loading.",
            "kinovea_trunk_ref": "70°-95° (Deep forward bending while sweeping)",
            "kinovea_elbow_ref": "50°-90° (Sweeping & drum loading)",
            "freq_ref": "20-30 sweeping actions/min"
        },
        "Crop Transport / Load Carrying": {
            "desc": "Manual carrying of harvested crop crates, bags, or buckets.",
            "hazard": "L5/S1 lumbar disc compression & excessive cardiac load.",
            "kinovea_trunk_ref": "0°-20° (Upright carrying)",
            "kinovea_elbow_ref": "40°-80° (Holding)",
            "freq_ref": "N/A"
        },
        "General Agricultural Work": {
            "desc": "Mixed field activity including walking, standing, and light handling.",
            "hazard": "General physical fatigue.",
            "kinovea_trunk_ref": "0°-30°",
            "kinovea_elbow_ref": "90°-140°",
            "freq_ref": "N/A"
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
        Classify field activity task calibrated with Turmeric Cultivation ground truth.
        """
        bending_pct = posture_distribution.get("bending", 0.0)
        squatting_pct = posture_distribution.get("squatting", 0.0)
        sitting_pct = posture_distribution.get("sitting", 0.0)
        standing_pct = posture_distribution.get("standing", 0.0)
        walking_pct = posture_distribution.get("walking", 0.0)
        cpm = cycles_per_minute or 0.0

        # 1. Crop Transport / Load Carrying
        if total_load_events > 0 or "backpack" in detected_tools or "suitcase" in detected_tools:
            if walking_pct > 15.0 or standing_pct > 30.0:
                info = self.TASKS["Crop Transport / Load Carrying"]
                return TaskClassificationResult(
                    primary_task="Crop Transport / Load Carrying",
                    confidence=0.88,
                    description=info["desc"],
                    ergonomic_hazard_profile=info["hazard"],
                )

        # 2. Turmeric Cleaning & Rhizome Separation (Fast finger/hand movements 40-55/min while sitting/squatting)
        if (squatting_pct + sitting_pct) > 25.0 and cpm >= 40.0 and repetitive_joint in ["wrist", "elbow"]:
            info = self.TASKS["Cleaning & Rhizome Separation"]
            return TaskClassificationResult(
                primary_task="Cleaning & Rhizome Separation",
                confidence=0.92,
                description=info["desc"],
                ergonomic_hazard_profile=info["hazard"],
            )

        # 3. Turmeric Seed Rhizome Prep / Sorting & Grading (Sitting/Squatting 30-50/min)
        if (squatting_pct + sitting_pct + bending_pct) > 35.0 and 30.0 <= cpm < 55.0:
            if sitting_pct > 20.0 or (squatting_pct > 20.0 and cpm >= 35.0):
                info = self.TASKS["Sorting & Grading"]
                return TaskClassificationResult(
                    primary_task="Sorting & Grading",
                    confidence=0.88,
                    description=info["desc"],
                    ergonomic_hazard_profile=info["hazard"],
                )
            info = self.TASKS["Seed Rhizome Preparation & Treatment"]
            return TaskClassificationResult(
                primary_task="Seed Rhizome Preparation & Treatment",
                confidence=0.86,
                description=info["desc"],
                ergonomic_hazard_profile=info["hazard"],
            )

        # 4. Sowing & Planting (Squatting 15-25 placements/min)
        if (squatting_pct > 25.0 or (squatting_pct + bending_pct > 40.0)) and 12.0 <= cpm <= 28.0:
            info = self.TASKS["Sowing & Rhizome Planting"]
            return TaskClassificationResult(
                primary_task="Sowing & Rhizome Planting",
                confidence=0.90,
                description=info["desc"],
                ergonomic_hazard_profile=info["hazard"],
            )

        # 5. Digging & Rhizome Harvesting (Deep squatting/kneeling + bending with trowel)
        if (squatting_pct + bending_pct) > 35.0 and ("trowel" in detected_tools or cpm < 25.0):
            info = self.TASKS["Digging & Rhizome Harvesting"]
            return TaskClassificationResult(
                primary_task="Digging & Rhizome Harvesting",
                confidence=0.87,
                description=info["desc"],
                ergonomic_hazard_profile=info["hazard"],
            )

        # 6. Land & Bed Preparation (Hoeing / Ridging) (Bending/Walking, 25-35 strokes/min trunk/elbow)
        if (bending_pct > 10.0 or walking_pct > 30.0) and cpm >= 20.0 and repetitive_joint in ["trunk", "elbow", "shoulder"]:
            info = self.TASKS["Land & Bed Preparation (Hoeing / Ridging)"]
            return TaskClassificationResult(
                primary_task="Land & Bed Preparation (Hoeing / Ridging)",
                confidence=0.89,
                description=info["desc"],
                ergonomic_hazard_profile=info["hazard"],
            )

        # 7. Boiling / Parboiling (Standing upright/slight lean 18-28 strokes/min)
        if standing_pct > 60.0 and 15.0 <= cpm <= 30.0 and repetitive_joint in ["elbow", "shoulder"]:
            info = self.TASKS["Boiling / Parboiling"]
            return TaskClassificationResult(
                primary_task="Boiling / Parboiling",
                confidence=0.84,
                description=info["desc"],
                ergonomic_hazard_profile=info["hazard"],
            )

        # 8. Polishing (Deep forward bending >70°, 20-30/min)
        if bending_pct > 25.0 and standing_pct > 25.0:
            info = self.TASKS["Polishing (Mechanical / Drum)"]
            return TaskClassificationResult(
                primary_task="Polishing (Mechanical / Drum)",
                confidence=0.80,
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
