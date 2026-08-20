"""
NIOSH Lifting Equation & L5/S1 Spinal Compression Force Engine.

Calculates:
1. Recommended Weight Limit (RWL) per NIOSH 1991 Manual Lifting Equation:
   RWL = LC x HM x VM x DM x AM x FM x CM
   Where LC = 23 kg (50 lbs) load constant.
2. Lifting Index (LI) = Actual Load Weight / RWL
   LI > 1.0 indicates increased risk of lower back injury.
3. L5/S1 Lumbar Spinal Compression Force (in Newtons):
   Estimates disc compression force based on trunk flexion angle, upper body mass, and load distance.
   Flags forces exceeding the 3,400 N (3.4 kN) NIOSH action limit.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple
from pathlib import Path


@dataclass
class NIOSHResult:
    """Result of NIOSH Lifting Equation and Lumbar Compression calculation."""
    recommended_weight_limit_kg: float   # Recommended Weight Limit (RWL)
    lifting_index: float                # Actual Load / RWL
    l5s1_compression_force_n: float     # Estimated L5/S1 spinal compression force (Newtons)
    is_compression_exceeded: bool       # True if compression > 3400 N
    risk_assessment: str                # Safe / Moderate Risk / High Risk of Back Injury
    recommendation: str                 # Ergonomic recommendation


class NIOSHCalculator:
    """
    Biomechanical Lumbar Stress and NIOSH Lifting Index Calculator.
    """

    LOAD_CONSTANT_KG = 23.0             # Standard NIOSH Load Constant (23 kg)
    NIOSH_SAFETY_COMPRESSION_N = 3400.0 # 3.4 kN spinal compression limit

    def calculate(
        self,
        actual_weight_kg: float = 5.0,
        trunk_flexion_degrees: float = 0.0,
        horizontal_distance_cm: float = 30.0,
        vertical_height_cm: float = 75.0,
        vertical_travel_cm: float = 50.0,
        asymmetry_angle_degrees: float = 0.0,
        lifts_per_minute: float = 1.0,
    ) -> NIOSHResult:
        """
        Calculate RWL, Lifting Index, and L5/S1 Spinal Compression Force.

        Args:
            actual_weight_kg: Carried load weight in kg.
            trunk_flexion_degrees: Forward bending angle from vertical.
            horizontal_distance_cm: Distance of load from mid-point between ankles.
            vertical_height_cm: Initial height of load from floor.
            vertical_travel_cm: Vertical displacement during lift.
            asymmetry_angle_degrees: Twisting angle from sagittal plane.
            lifts_per_minute: Frequency of lifts per minute.

        Returns:
            NIOSHResult object.
        """
        actual_weight_kg = max(0.5, actual_weight_kg)
        trunk_flexion_degrees = max(0.0, trunk_flexion_degrees)

        # ── 1. NIOSH Multipliers ──
        # Horizontal Multiplier (HM = 25 / H)
        h = max(25.0, min(63.0, horizontal_distance_cm))
        hm = 25.0 / h

        # Vertical Multiplier (VM = 1 - 0.003 |V - 75|)
        v = max(0.0, min(175.0, vertical_height_cm))
        vm = 1.0 - (0.003 * abs(v - 75.0))

        # Distance Multiplier (DM = 0.82 + 4.5 / D)
        d = max(25.0, min(175.0, vertical_travel_cm))
        dm = 0.82 + (4.5 / d)

        # Asymmetry Multiplier (AM = 1 - 0.0032 A)
        a = max(0.0, min(135.0, asymmetry_angle_degrees))
        am = 1.0 - (0.0032 * a)

        # Frequency Multiplier (FM)
        if lifts_per_minute <= 0.2:
            fm = 1.00
        elif lifts_per_minute <= 1.0:
            fm = 0.94
        elif lifts_per_minute <= 4.0:
            fm = 0.84
        else:
            fm = 0.72

        # Coupling Multiplier (CM) — Fair grip default
        cm = 0.95

        # ── 2. Recommended Weight Limit (RWL) ──
        rwl = float(round(self.LOAD_CONSTANT_KG * hm * vm * dm * am * fm * cm, 2))
        rwl = max(0.5, rwl)

        # ── 3. Lifting Index (LI) ──
        li = float(round(actual_weight_kg / rwl, 2))

        # ── 4. L5/S1 Lumbar Compression Force Estimation (Newtons) ──
        # Biomechanical model:
        # F_comp = (Upper Body Mass x g x sin(trunk_flexion) x moment_arm_trunk + Load x g x moment_arm_load) / Erector_Spinae_Moment_Arm
        # Upper body mass ~ 60% of 70kg worker = 42 kg
        # Erector spinae muscle moment arm ~ 5 cm (0.05 m)
        g = 9.81
        m_upper = 42.0  # kg
        rad_flexion = np.radians(trunk_flexion_degrees)

        # Moment arms in meters
        d_trunk = 0.20 * np.sin(rad_flexion) + 0.05  # m
        d_load = (horizontal_distance_cm / 100.0) * np.sin(rad_flexion) + 0.15  # m
        d_muscle = 0.05  # m

        torque_trunk = m_upper * g * d_trunk
        torque_load = actual_weight_kg * g * d_load
        f_muscle = (torque_trunk + torque_load) / d_muscle

        # Total compressive force on L5/S1 disc (muscle force + body weight force component)
        f_compression = float(round(f_muscle + (m_upper + actual_weight_kg) * g * np.cos(rad_flexion), 1))

        is_exceeded = bool(f_compression > self.NIOSH_SAFETY_COMPRESSION_N)

        # ── 5. Risk Assessment Mapping ──
        if li <= 1.0 and not is_exceeded:
            risk = "Low Risk (Safe)"
            rec = "Lifting task is within safe NIOSH biomechanical guidelines."
        elif li <= 2.0 and f_compression <= 4000:
            risk = "Moderate Risk"
            rec = "Reduce load distance or reduce forward trunk flexion to protect lumbar spine."
        else:
            risk = "High Risk (Severe Spinal Strain)"
            rec = "CRITICAL: Spinal compression exceeds safe 3.4 kN limit! Implement mechanical lifting assistance or redesign task."

        return NIOSHResult(
            recommended_weight_limit_kg=rwl,
            lifting_index=li,
            l5s1_compression_force_n=f_compression,
            is_compression_exceeded=is_exceeded,
            risk_assessment=risk,
            recommendation=rec,
        )
