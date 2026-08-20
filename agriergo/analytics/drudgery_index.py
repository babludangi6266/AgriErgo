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
    drudgery_category: str              # Low / Moderate / High / Severe Drudgery
    postural_stress_score: float        # 0–100
    repetitive_strain_score: float      # 0–100
    work_rest_strain_score: float       # 0–100
    load_strain_score: float            # 0–100
    estimated_fatigue_level: float      # 0–100 % accumulated fatigue
    recommendations: List[str]          # Actionable intervention recommendations
    minute_fatigue_series: List[float] = field(default_factory=list) # Minute-by-minute fatigue values (%)


class DrudgeryCalculator:
    """
    Calculates Agricultural Drudgery Index (ADI).
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
    ) -> DrudgeryResult:
        """
        Compute the Agricultural Drudgery Index (ADI).
        """
        # 1. Postural Stress (0-100)
        # Weighted by high-risk postures (bending, squatting) and ergonomic scores
        bending_pct = posture_distribution.get("bending", 0.0)
        squatting_pct = posture_distribution.get("squatting", 0.0)
        
        postural_raw = (
            (reba_score / 15.0 * 40.0) +
            (rula_score / 7.0 * 30.0) +
            (bending_pct * 0.20) +
            (squatting_pct * 0.10)
        )
        postural_stress = float(np.clip(postural_raw, 0.0, 100.0))

        # 2. Repetitive Strain (0-100)
        rep_cpm = cycles_per_minute or 0.0
        if rep_cpm < 10:
            rep_strain = 10.0
        elif rep_cpm < 30:
            rep_strain = 35.0
        elif rep_cpm < 60:
            rep_strain = 65.0
        else:
            rep_strain = 95.0

        # 3. Work-Rest Ratio Strain (0-100)
        rest_pct = (total_rest_seconds / max(1.0, total_tracked_seconds)) * 100.0
        work_bout_mins = longest_work_bout_seconds / 60.0
        
        work_rest_raw = (work_bout_mins * 2.5) + (100.0 - min(100.0, rest_pct * 2.0))
        work_rest_strain = float(np.clip(work_rest_raw, 0.0, 100.0))

        # 4. Load Strain (0-100)
        load_strain = float(np.clip(load_events_count * 15.0, 0.0, 100.0))

        # Combined Agricultural Drudgery Index (weighted composite)
        adi = (
            (postural_stress * 0.40) +
            (rep_strain * 0.25) +
            (work_rest_strain * 0.20) +
            (load_strain * 0.15)
        )
        adi = float(round(np.clip(adi, 0.0, 100.0), 1))

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

        # Minute-by-minute fatigue curve over tracked session (e.g. 1 to 10 minutes)
        total_mins = max(1, int(np.ceil(total_tracked_seconds / 60.0)))
        minute_fatigue_series = []
        for m in range(1, total_mins + 1):
            # Accumulation curve formula with asymptotic saturation at overall fatigue level
            progress = m / total_mins
            f_m = float(round(fatigue * (1.0 - np.exp(-2.5 * progress)), 1))
            minute_fatigue_series.append(f_m)

        # Generate Actionable Recommendations
        recs = []
        if postural_stress > 50:
            recs.append("Implement long-handled tools to reduce trunk stooping/bending.")
        if rep_strain > 50:
            recs.append("Introduce micro-break cycles (1 min rest every 15 mins of repetitive motion).")
        if work_rest_strain > 50:
            recs.append("Increase scheduled rest duration between long work bouts.")
        if load_strain > 30:
            recs.append("Use wheelbarrows or mechanical carrying aids to reduce manual load stress.")
        if not recs:
            recs.append("Current work setup and posture distribution are within safe ergonomic limits.")

        return DrudgeryResult(
            drudgery_index=adi,
            drudgery_category=category,
            postural_stress_score=round(postural_stress, 1),
            repetitive_strain_score=round(rep_strain, 1),
            work_rest_strain_score=round(work_rest_strain, 1),
            load_strain_score=round(load_strain, 1),
            estimated_fatigue_level=fatigue,
            recommendations=recs,
            minute_fatigue_series=minute_fatigue_series,
        )
