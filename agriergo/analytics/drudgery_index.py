"""
Agricultural Drudgery Index (ADI) & Fatigue Tracker.

Calculates a standardized Agricultural Drudgery Index (0–100 scale) combining:
1. Postural Risk Score (REBA/RULA distribution)
2. Repetitive Motion Strain Factor
3. Work-Rest Ratio & Continuous Work Bout Duration
4. Carried Load & Force Factor

Also computes cumulative fatigue curves over the work session.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


@dataclass
class DrudgeryResult:
    """Agricultural Drudgery Index calculation result."""
    drudgery_index: float                # 0–100 score
    drudgery_percentage: float           # 0-100 % drudgery index
    drudgery_category: str              # Low / Moderate / High / Severe Drudgery
    postural_stress_score: float        # 0–100
    repetitive_strain_score: float      # 0–100
    work_rest_strain_score: float       # 0–100
    load_strain_score: float            # 0–100
    arm_strain_score: float = 0.0       # 0–100
    estimated_fatigue_level: float = 0.0 # 0–100 % accumulated fatigue
    recommendations: List[str] = field(default_factory=list) # Actionable intervention recommendations
    minute_fatigue_series: List[float] = field(default_factory=list) # Minute-by-minute fatigue values (%)


class DrudgeryCalculator:
    """
    Calculates Agricultural Drudgery Index (ADI) and Percentage.
    """

    def calculate(
        self,
        reba_score: float,
        rula_score: float,
        posture_distribution: Dict[str, float],   # posture_name -> percentage
        cycles_per_minute: Optional[float],
        total_tracked_seconds: float,
        longest_work_bout_seconds: float,
        total_rest_seconds: float,
        load_events_count: int,
        severe_bending_pct: float = 0.0,
        shoulder_above_45_pct: float = 0.0,
        shoulder_above_90_pct: float = 0.0,
    ) -> DrudgeryResult:
        """
        Compute the Agricultural Drudgery Index (ADI) and Drudgery Percentage.
        """
        # 1. Postural Stress (0-100)
        # Weighted by high-risk postures (bending, severe bending, squatting) and ergonomic scores
        bending_pct = posture_distribution.get("bending", 0.0)
        squatting_pct = posture_distribution.get("squatting", 0.0)
        
        postural_raw = (
            (reba_score / 15.0 * 35.0) +
            (rula_score / 7.0 * 25.0) +
            (bending_pct * 0.20) +
            (severe_bending_pct * 0.15) +
            (squatting_pct * 0.10)
        )
        postural_stress = float(np.clip(postural_raw, 0.0, 100.0))

        # 2. Repetitive Strain (0-100 continuous curve)
        rep_cpm = cycles_per_minute or 0.0
        if rep_cpm > 0:
            # Smooth sigmoid centered at 30 CPM
            rep_strain = float(100.0 / (1.0 + np.exp(-0.08 * (rep_cpm - 25.0))))
        else:
            rep_strain = 10.0
        rep_strain = float(np.clip(rep_strain, 0.0, 100.0))

        # 3. Work-Rest Ratio Strain (0-100)
        rest_pct = (total_rest_seconds / max(1.0, total_tracked_seconds)) * 100.0
        work_bout_mins = longest_work_bout_seconds / 60.0
        
        work_rest_raw = (work_bout_mins * 3.0) + (100.0 - min(100.0, rest_pct * 2.5))
        work_rest_strain = float(np.clip(work_rest_raw, 0.0, 100.0))

        # 4. Load Strain (0-100)
        load_strain = float(np.clip(load_events_count * 15.0, 0.0, 100.0))

        # 5. Arm Postural Strain (0-100) — Overhead and elevated arm work
        arm_raw = (shoulder_above_45_pct * 0.6) + (shoulder_above_90_pct * 1.2)
        arm_strain = float(np.clip(arm_raw, 0.0, 100.0))

        # Combined Agricultural Drudgery Index (weighted composite across 5 pillars)
        adi = (
            (postural_stress * 0.35) +
            (rep_strain * 0.20) +
            (work_rest_strain * 0.20) +
            (load_strain * 0.10) +
            (arm_strain * 0.15)
        )
        adi = float(round(np.clip(adi, 0.0, 100.0), 1))
        drudgery_pct = adi  # Direct percentage equivalent

        # Category mapping
        if adi < 30.0:
            category = "Low Drudgery"
        elif adi < 55.0:
            category = "Moderate Drudgery"
        elif adi < 75.0:
            category = "High Drudgery"
        else:
            category = "Severe Drudgery"

        # Fatigue level estimation
        fatigue = float(round(np.clip(adi * 0.85 + (work_bout_mins * 0.5), 0.0, 100.0), 1))

        # Minute-by-minute fatigue curve over tracked session
        total_mins = max(1, int(np.ceil(total_tracked_seconds / 60.0)))
        minute_fatigue_series = []
        for m in range(1, total_mins + 1):
            progress = m / total_mins
            f_m = float(round(fatigue * (1.0 - np.exp(-2.5 * progress)), 1))
            minute_fatigue_series.append(f_m)

        # Generate Actionable Recommendations
        recs = []
        if arm_strain > 40:
            recs.append("Reduce overhead reaching and elevated arm posture using adjustable trestles or step platforms.")
        if postural_stress > 50:
            recs.append("Implement long-handled tools to reduce trunk stooping/bending.")
        if rep_strain > 50:
            recs.append("Introduce micro-break cycles (1 min rest every 15 mins of repetitive motion).")
        if work_rest_strain > 50:
            recs.append("Increase scheduled rest duration between long continuous work bouts.")
        if load_strain > 30:
            recs.append("Use wheelbarrows or mechanical carrying aids to reduce manual load stress.")
        if not recs:
            recs.append("Current work setup and posture distribution are within safe ergonomic limits.")

        return DrudgeryResult(
            drudgery_index=adi,
            drudgery_percentage=drudgery_pct,
            drudgery_category=category,
            postural_stress_score=round(postural_stress, 1),
            repetitive_strain_score=round(rep_strain, 1),
            work_rest_strain_score=round(work_rest_strain, 1),
            load_strain_score=round(load_strain, 1),
            arm_strain_score=round(arm_strain, 1),
            estimated_fatigue_level=fatigue,
            recommendations=recs,
            minute_fatigue_series=minute_fatigue_series,
        )
