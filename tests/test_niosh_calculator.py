"""
Unit tests for NIOSH Lifting Equation & L5/S1 Spinal Compression Calculator.
"""

import pytest
from agriergo.analytics.niosh_calculator import NIOSHCalculator


def test_niosh_calculator_safe():
    calc = NIOSHCalculator()
    res = calc.calculate(
        actual_weight_kg=5.0,
        trunk_flexion_degrees=10.0,
        horizontal_distance_cm=30.0,
    )
    assert res.recommended_weight_limit_kg > 0.0
    assert res.lifting_index > 0.0
    assert res.l5s1_compression_force_n > 0.0
    assert not res.is_compression_exceeded
    assert "Low Risk" in res.risk_assessment or "Safe" in res.risk_assessment


def test_niosh_calculator_high_risk():
    calc = NIOSHCalculator()
    res = calc.calculate(
        actual_weight_kg=25.0,
        trunk_flexion_degrees=65.0,
        horizontal_distance_cm=55.0,
    )
    assert res.lifting_index > 1.0
    assert res.l5s1_compression_force_n > 3400.0
    assert res.is_compression_exceeded
    assert "High Risk" in res.risk_assessment
