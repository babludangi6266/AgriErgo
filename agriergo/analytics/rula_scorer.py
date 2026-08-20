"""
RULA Scorer — Rapid Upper Limb Assessment for Agricultural Tasks.

Calculates RULA ergonomic risk scores (1-7) specifically designed for tasks
involving repetitive upper body work (weeding, harvesting, picking, pruning, sickle cutting).

RULA Structure:
- Group A: Upper Arm, Lower Arm, Wrist, Wrist Twist
- Group B: Neck, Trunk, Legs
- Muscle Use & Load Adjustments
- Final RULA Score (1-7) & Action Levels
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from agriergo.interpretation.joint_angles import JointAngles


# ──────────────────────────────────────────────
# RULA Scoring Tables
# ──────────────────────────────────────────────

# Table A: Upper Arm x Lower Arm x Wrist x Wrist Twist -> Score A
# Format: Upper Arm (1-6) x Lower Arm (1-3) x Wrist (1-4)
RULA_TABLE_A = [
    # Upper Arm = 1
    [[1, 2, 2, 3], [2, 2, 3, 3], [2, 3, 3, 4]],
    # Upper Arm = 2
    [[2, 3, 3, 4], [3, 3, 4, 4], [3, 4, 4, 5]],
    # Upper Arm = 3
    [[3, 3, 4, 4], [4, 4, 4, 5], [4, 4, 5, 5]],
    # Upper Arm = 4
    [[4, 4, 4, 5], [4, 4, 5, 5], [4, 5, 5, 6]],
    # Upper Arm = 5
    [[5, 5, 5, 6], [5, 6, 6, 7], [6, 6, 7, 7]],
    # Upper Arm = 6
    [[7, 7, 7, 8], [8, 8, 8, 9], [9, 9, 9, 9]],
]

# Table B: Neck x Trunk x Legs -> Score B
# Neck (1-6) x Trunk (1-6) x Legs (1-2)
RULA_TABLE_B = [
    # Neck = 1
    [[1, 3], [2, 3], [3, 4], [5, 5], [6, 6], [7, 7]],
    # Neck = 2
    [[2, 3], [3, 4], [4, 5], [5, 5], [6, 7], [7, 7]],
    # Neck = 3
    [[3, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 7]],
    # Neck = 4
    [[5, 5], [5, 6], [6, 7], [7, 7], [7, 7], [8, 8]],
    # Neck = 5
    [[7, 7], [7, 7], [7, 8], [8, 8], [8, 8], [8, 8]],
    # Neck = 6
    [[8, 8], [8, 8], [8, 8], [8, 9], [9, 9], [9, 9]],
]

# Table C: Score A (1-8) x Score B (1-7) -> Final RULA Score (1-7)
RULA_TABLE_C = [
    #  B=1  2  3  4  5  6  7+
    [1, 2, 3, 3, 4, 5, 5],  # A=1
    [2, 2, 3, 4, 4, 5, 5],  # A=2
    [3, 3, 3, 4, 4, 5, 6],  # A=3
    [3, 3, 3, 4, 5, 6, 6],  # A=4
    [4, 4, 4, 5, 6, 7, 7],  # A=5
    [4, 4, 5, 6, 6, 7, 7],  # A=6
    [5, 5, 6, 6, 7, 7, 7],  # A=7
    [5, 5, 6, 7, 7, 7, 7],  # A=8+
]

RULA_ACTION_LEVELS = [
    (1, 2, "Acceptable", "Posture is acceptable if not maintained/repeated for long periods"),
    (3, 4, "Investigate Further", "Further investigation is needed and changes may be required"),
    (5, 6, "Investigate & Change Soon", "Investigation and changes are required soon"),
    (7, 7, "Investigate & Change Immediately", "Investigation and changes are required IMMEDIATELY"),
]


@dataclass
class RULAScore:
    """Complete RULA score breakdown."""
    upper_arm_score: int = 1
    lower_arm_score: int = 1
    wrist_score: int = 1
    wrist_twist_score: int = 1
    score_a: int = 1

    neck_score: int = 1
    trunk_score: int = 1
    legs_score: int = 1
    score_b: int = 1

    muscle_score: int = 0      # 1 if posture is static (>1 min) or repeated (>4x/min)
    load_score: int = 0        # 0-3 based on weight/force

    score_a_total: int = 1
    score_b_total: int = 1
    final_score: int = 1
    action_level: str = "Acceptable"
    action_recommendation: str = "Acceptable posture"


class RULAScorer:
    """
    Computes RULA ergonomic risk scores from joint angle data.
    """

    def score_frame(
        self,
        joint_angles: JointAngles,
        is_repetitive: bool = False,
        load_force: int = 0,
    ) -> RULAScore:
        """
        Compute RULA score for a single frame.

        Args:
            joint_angles: Computed joint angles.
            is_repetitive: Action repeated >4 times per minute.
            load_force: Carried load / force score (0-3).

        Returns:
            Complete RULAScore.
        """
        rula = RULAScore()

        # ── 1. Upper Arm Score ──
        # Shoulder elevation / arm flexion angle
        arm_angle = None
        angles = [joint_angles.left_shoulder_angle, joint_angles.right_shoulder_angle]
        valid_angles = [a for a in angles if a is not None]
        if valid_angles:
            arm_angle = max(valid_angles)

        if arm_angle is None:
            rula.upper_arm_score = 1
        elif arm_angle < 20:
            rula.upper_arm_score = 1
        elif arm_angle < 45:
            rula.upper_arm_score = 2
        elif arm_angle < 90:
            rula.upper_arm_score = 3
        else:
            rula.upper_arm_score = 4

        # ── 2. Lower Arm Score ──
        # Elbow angle (neutral is 60°-100°)
        elbow_angle = joint_angles.left_elbow_angle or joint_angles.right_elbow_angle
        if elbow_angle is None:
            rula.lower_arm_score = 1
        elif 60 <= elbow_angle <= 100:
            rula.lower_arm_score = 1
        else:
            rula.lower_arm_score = 2

        # ── 3. Wrist Score ──
        wrist_angle = joint_angles.left_wrist_angle or joint_angles.right_wrist_angle
        if wrist_angle is None:
            rula.wrist_score = 1
        elif abs(90 - wrist_angle) < 15:
            rula.wrist_score = 1
        else:
            rula.wrist_score = 2

        rula.wrist_twist_score = 1

        # ── Group A Score ──
        rula.score_a = self._lookup_table_a(
            rula.upper_arm_score, rula.lower_arm_score, rula.wrist_score
        )

        # ── 4. Neck Score ──
        neck_flex = joint_angles.neck_flexion
        if neck_flex is None:
            rula.neck_score = 1
        elif neck_flex < 10:
            rula.neck_score = 1
        elif neck_flex < 20:
            rula.neck_score = 2
        else:
            rula.neck_score = 3

        # ── 5. Trunk Score ──
        trunk_flex = joint_angles.trunk_flexion
        if trunk_flex is None:
            rula.trunk_score = 1
        elif trunk_flex < 5:
            rula.trunk_score = 1
        elif trunk_flex < 20:
            rula.trunk_score = 2
        elif trunk_flex < 60:
            rula.trunk_score = 3
        else:
            rula.trunk_score = 4

        # ── 6. Legs Score ──
        # 1 if supported/walking, 2 if unsupported/squatting
        knee_angle = joint_angles.avg_knee_angle
        if knee_angle is not None and knee_angle < 110:
            rula.legs_score = 2
        else:
            rula.legs_score = 1

        # ── Group B Score ──
        rula.score_b = self._lookup_table_b(
            rula.neck_score, rula.trunk_score, rula.legs_score
        )

        # ── Adjustments ──
        rula.muscle_score = 1 if is_repetitive else 0
        rula.load_score = min(3, max(0, load_force))

        rula.score_a_total = rula.score_a + rula.muscle_score + rula.load_score
        rula.score_b_total = rula.score_b + rula.muscle_score + rula.load_score

        # ── Final Score via Table C ──
        rula.final_score = self._lookup_table_c(rula.score_a_total, rula.score_b_total)

        # Map to action level
        rula.action_level, rula.action_recommendation = self._get_action_level(
            rula.final_score
        )

        return rula

    def score_worker_overall(self, frame_scores: List[RULAScore]) -> RULAScore:
        """Compute 90th percentile overall RULA score for a worker."""
        if not frame_scores:
            return RULAScore()

        scores = [s.final_score for s in frame_scores]
        p90 = int(np.percentile(scores, 90))
        p90 = max(1, min(7, p90))

        closest = min(frame_scores, key=lambda s: abs(s.final_score - p90))
        overall = RULAScore(
            upper_arm_score=closest.upper_arm_score,
            lower_arm_score=closest.lower_arm_score,
            wrist_score=closest.wrist_score,
            score_a=closest.score_a,
            neck_score=closest.neck_score,
            trunk_score=closest.trunk_score,
            legs_score=closest.legs_score,
            score_b=closest.score_b,
            muscle_score=closest.muscle_score,
            load_score=closest.load_score,
            score_a_total=closest.score_a_total,
            score_b_total=closest.score_b_total,
            final_score=p90,
        )
        overall.action_level, overall.action_recommendation = self._get_action_level(p90)
        return overall

    def _lookup_table_a(self, upper_arm: int, lower_arm: int, wrist: int) -> int:
        ua = min(upper_arm, 6) - 1
        la = min(lower_arm, 3) - 1
        w = min(wrist, 4) - 1
        try:
            return RULA_TABLE_A[max(0, ua)][max(0, la)][max(0, w)]
        except IndexError:
            return 1

    def _lookup_table_b(self, neck: int, trunk: int, legs: int) -> int:
        n = min(neck, 6) - 1
        t = min(trunk, 6) - 1
        l = min(legs, 2) - 1
        try:
            return RULA_TABLE_B[max(0, n)][max(0, t)][max(0, l)]
        except IndexError:
            return 1

    def _lookup_table_c(self, score_a: int, score_b: int) -> int:
        sa = min(score_a, 8) - 1
        sb = min(score_b, 7) - 1
        try:
            return RULA_TABLE_C[max(0, sa)][max(0, sb)]
        except IndexError:
            return 1

    def _get_action_level(self, score: int) -> Tuple[str, str]:
        for min_s, max_s, lvl, rec in RULA_ACTION_LEVELS:
            if min_s <= score <= max_s:
                return lvl, rec
        return "Investigate & Change Immediately", "High risk posture"
