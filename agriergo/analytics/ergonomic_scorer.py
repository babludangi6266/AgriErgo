"""
Ergonomic Scorer — REBA-based automated ergonomic risk scoring.

Implements the Rapid Entire Body Assessment (REBA) scoring methodology
using joint angles derived from pose estimation. Produces a risk score
(1-15) and risk level for each work bout and overall per worker.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    REBA_TRUNK_SCORES,
    REBA_NECK_SCORES,
    REBA_LEGS_SCORES,
    REBA_UPPER_ARM_SCORES,
    REBA_LOWER_ARM_SCORES,
    REBA_WRIST_SCORES,
    REBA_TABLE_A,
    REBA_TABLE_B,
    REBA_TABLE_C,
    REBA_RISK_LEVELS,
)
from agriergo.interpretation.joint_angles import JointAngles


@dataclass
class REBAScore:
    """Complete REBA scoring breakdown."""
    # Individual body part scores
    trunk_score: int = 1
    neck_score: int = 1
    legs_score: int = 1
    upper_arm_score: int = 1
    lower_arm_score: int = 1
    wrist_score: int = 1

    # Group scores
    score_a: int = 1            # From Table A (trunk, neck, legs)
    score_b: int = 1            # From Table B (upper arm, lower arm, wrist)

    # Load/force adjustment
    load_force_score: int = 0   # 0-3

    # Activity adjustment
    activity_score: int = 0     # 0-3

    # Final scores
    score_a_adjusted: int = 1   # Score A + load/force
    score_b_adjusted: int = 1   # Score B + coupling (not used in prototype)
    score_c: int = 1            # From Table C
    final_score: int = 1        # Score C + activity
    risk_level: str = "Negligible"
    action_required: str = "None necessary"


class ErgonomicScorer:
    """
    Computes REBA ergonomic risk scores from joint angle data.

    REBA Process:
    1. Score individual body segments based on joint angles
    2. Combine into Group A (trunk, neck, legs) and Group B (arms, wrists)
    3. Look up combined scores in Tables A, B, C
    4. Add adjustments for load/force and activity
    5. Map final score to risk level

    For the prototype, load/force is set via manual input (default 0),
    and activity score is derived from posture data.
    """

    def score_frame(
        self,
        joint_angles: JointAngles,
        load_force: int = 0,
        is_static: bool = False,
        is_repetitive: bool = False,
        has_rapid_changes: bool = False,
    ) -> REBAScore:
        """
        Compute REBA score for a single frame's joint angles.

        Args:
            joint_angles: Computed joint angles.
            load_force: Load/force score (0-3).
            is_static: Held for >1 minute.
            is_repetitive: Repeated more than 4x/minute.
            has_rapid_changes: Rapid large changes in posture.

        Returns:
            Complete REBAScore breakdown.
        """
        reba = REBAScore()

        # ── Score individual body segments ──

        # Trunk
        reba.trunk_score = self._score_from_ranges(
            joint_angles.trunk_flexion, REBA_TRUNK_SCORES, default=2
        )

        # Neck
        reba.neck_score = self._score_from_ranges(
            joint_angles.neck_flexion, REBA_NECK_SCORES, default=1
        )

        # Legs (based on knee flexion — deviation from 180° straight)
        knee_flexion = None
        if joint_angles.avg_knee_angle is not None:
            knee_flexion = abs(180 - joint_angles.avg_knee_angle)
        reba.legs_score = self._score_from_ranges(
            knee_flexion, REBA_LEGS_SCORES, default=1
        )

        # Upper arm (shoulder angle / arm elevation)
        shoulder_angle = None
        angles = [joint_angles.left_shoulder_angle, joint_angles.right_shoulder_angle]
        valid_angles = [a for a in angles if a is not None]
        if valid_angles:
            shoulder_angle = max(valid_angles)  # Use the worse side
        reba.upper_arm_score = self._score_from_ranges(
            shoulder_angle, REBA_UPPER_ARM_SCORES, default=1
        )

        # Lower arm (elbow angle)
        elbow_angle = None
        elbow_angles = [joint_angles.left_elbow_angle, joint_angles.right_elbow_angle]
        valid_elbows = [a for a in elbow_angles if a is not None]
        if valid_elbows:
            elbow_angle = np.mean(valid_elbows)
        reba.lower_arm_score = self._score_from_ranges(
            elbow_angle, REBA_LOWER_ARM_SCORES, default=1
        )

        # Wrist
        wrist_angle = None
        wrist_angles = [joint_angles.left_wrist_angle, joint_angles.right_wrist_angle]
        valid_wrists = [a for a in wrist_angles if a is not None]
        if valid_wrists:
            # Convert from angle-from-vertical to deviation from neutral
            wrist_angle = abs(90 - np.mean(valid_wrists))  # Neutral is ~90° from vertical
        reba.wrist_score = self._score_from_ranges(
            wrist_angle, REBA_WRIST_SCORES, default=1
        )

        # ── Table A: trunk × neck × legs ──
        reba.score_a = self._lookup_table_a(
            reba.trunk_score, reba.neck_score, reba.legs_score
        )

        # ── Table B: upper arm × lower arm × wrist ──
        reba.score_b = self._lookup_table_b(
            reba.upper_arm_score, reba.lower_arm_score, reba.wrist_score
        )

        # ── Adjustments ──
        reba.load_force_score = min(3, max(0, load_force))
        reba.score_a_adjusted = reba.score_a + reba.load_force_score
        reba.score_b_adjusted = reba.score_b  # Coupling score omitted for prototype

        # ── Table C ──
        reba.score_c = self._lookup_table_c(
            reba.score_a_adjusted, reba.score_b_adjusted
        )

        # ── Activity score ──
        reba.activity_score = 0
        if is_static:
            reba.activity_score += 1
        if is_repetitive:
            reba.activity_score += 1
        if has_rapid_changes:
            reba.activity_score += 1

        # ── Final score ──
        reba.final_score = min(15, reba.score_c + reba.activity_score)

        # ── Risk level ──
        reba.risk_level, reba.action_required = self._get_risk_level(reba.final_score)

        return reba

    def score_worker_overall(
        self,
        frame_scores: List[REBAScore],
    ) -> REBAScore:
        """
        Compute overall REBA score for a worker from per-frame scores.

        Uses the 90th percentile of frame scores as the overall score
        (captures typical worst-case without being skewed by outliers).
        """
        if not frame_scores:
            return REBAScore()

        final_scores = [s.final_score for s in frame_scores]

        # Use 90th percentile
        p90_score = int(np.percentile(final_scores, 90))
        p90_score = max(1, min(15, p90_score))

        # Find the frame score closest to the P90 value
        closest = min(frame_scores, key=lambda s: abs(s.final_score - p90_score))

        # Create an overall score based on the P90 frame
        overall = REBAScore(
            trunk_score=closest.trunk_score,
            neck_score=closest.neck_score,
            legs_score=closest.legs_score,
            upper_arm_score=closest.upper_arm_score,
            lower_arm_score=closest.lower_arm_score,
            wrist_score=closest.wrist_score,
            score_a=closest.score_a,
            score_b=closest.score_b,
            load_force_score=closest.load_force_score,
            activity_score=closest.activity_score,
            score_a_adjusted=closest.score_a_adjusted,
            score_b_adjusted=closest.score_b_adjusted,
            score_c=closest.score_c,
            final_score=p90_score,
        )
        overall.risk_level, overall.action_required = self._get_risk_level(p90_score)

        return overall

    def _score_from_ranges(
        self,
        angle: Optional[float],
        ranges: List[Tuple],
        default: int = 1,
    ) -> int:
        """Look up score from angle ranges."""
        if angle is None:
            return default

        for min_angle, max_angle, score in ranges:
            if min_angle <= angle < max_angle:
                return score

        # If angle exceeds all ranges, return the last score
        return ranges[-1][2] if ranges else default

    def _lookup_table_a(self, trunk: int, neck: int, legs: int) -> int:
        """Look up Score A from REBA Table A."""
        t = min(trunk, 5) - 1
        n = min(neck, 2) - 1
        l = min(legs, 4) - 1

        t = max(0, t)
        n = max(0, n)
        l = max(0, l)

        try:
            return REBA_TABLE_A[t][n][l]
        except IndexError:
            return 1

    def _lookup_table_b(self, upper_arm: int, lower_arm: int, wrist: int) -> int:
        """Look up Score B from REBA Table B."""
        ua = min(upper_arm, 6) - 1
        la = min(lower_arm, 2) - 1
        w = min(wrist, 2) - 1

        ua = max(0, ua)
        la = max(0, la)
        w = max(0, w)

        try:
            return REBA_TABLE_B[ua][la][w]
        except IndexError:
            return 1

    def _lookup_table_c(self, score_a: int, score_b: int) -> int:
        """Look up final score from REBA Table C."""
        a = min(score_a, 12) - 1
        b = min(score_b, 12) - 1

        a = max(0, a)
        b = max(0, b)

        try:
            return REBA_TABLE_C[a][b]
        except IndexError:
            return 1

    def _get_risk_level(self, score: int) -> Tuple[str, str]:
        """Map REBA score to risk level and action recommendation."""
        for min_score, max_score, level, action in REBA_RISK_LEVELS:
            if min_score <= score <= max_score:
                return level, action
        return "Very High", "Necessary NOW"
