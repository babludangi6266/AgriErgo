"""
Tests for 3D Biomechanics & L5/S1 Spinal Compression Engine.
"""

import numpy as np
import pytest

from agriergo.interpretation.biomechanics_3d import (
    Point3D,
    Skeleton3D,
    SpineStress3D,
    compute_l5s1_spinal_compression,
    lift_2d_to_3d_skeleton,
    generate_3d_skeleton_plotly_figure,
)
from config.settings import NIOSH_COMPRESSION_ACTION_LIMIT


def test_point3d_as_array():
    """Test Point3D array conversion."""
    p = Point3D(1.0, 2.0, 3.0, confidence=0.9, name="knee")
    arr = p.as_array()
    assert np.allclose(arr, [1.0, 2.0, 3.0])
    assert p.confidence == 0.9


def test_l5s1_spinal_compression_upright():
    """Test spinal compression in neutral upright standing posture."""
    stress = compute_l5s1_spinal_compression(trunk_flexion_deg=0.0, load_kg=0.0)
    assert stress.l5s1_compression_n < 1500.0
    assert stress.l5s1_shear_n == 0.0
    assert stress.risk_level == "Safe"
    assert stress.color_hex == "#10B981"
    assert stress.action_limit_ratio < 1.0


def test_l5s1_spinal_compression_severe_stoop():
    """Test spinal compression during severe agricultural stooping (>60°)."""
    stress = compute_l5s1_spinal_compression(trunk_flexion_deg=65.0, load_kg=0.0)
    assert stress.l5s1_compression_n > 2000.0
    assert stress.l5s1_shear_n > 200.0
    assert stress.lumbar_stress > 0.5


def test_l5s1_spinal_compression_with_carried_load():
    """Test that manual crop load lifting exceeds NIOSH 3400 N action limit."""
    stress = compute_l5s1_spinal_compression(trunk_flexion_deg=60.0, load_kg=25.0)
    assert stress.l5s1_compression_n >= NIOSH_COMPRESSION_ACTION_LIMIT
    assert stress.risk_level in ["Action Limit Exceeded", "Hazardous"]
    assert stress.action_limit_ratio >= 1.0


def test_lift_2d_to_3d_skeleton():
    """Test 2.5D/3D kinematic lifting and spine node synthesis."""
    # Synthetic 17 COCO keypoints (x, y, conf)
    dummy_kp = np.zeros((17, 3), dtype=float)
    # Nose
    dummy_kp[0] = [320, 100, 0.9]
    # Shoulders
    dummy_kp[5] = [300, 180, 0.9]
    dummy_kp[6] = [340, 180, 0.9]
    # Hips
    dummy_kp[11] = [305, 300, 0.9]
    dummy_kp[12] = [335, 300, 0.9]
    # Knees
    dummy_kp[13] = [305, 420, 0.9]
    dummy_kp[14] = [335, 420, 0.9]
    # Ankles
    dummy_kp[15] = [305, 520, 0.9]
    dummy_kp[16] = [335, 520, 0.9]

    skel = lift_2d_to_3d_skeleton(dummy_kp, trunk_flexion_deg=45.0, knee_angle_deg=160.0)
    assert "pelvis" in skel.joints
    assert "lumbar_l5s1" in skel.joints
    assert "mid_spine_t12" in skel.joints
    assert "neck_c7" in skel.joints
    assert "head" in skel.joints

    # Forward depth (Z) should be positive due to forward trunk flexion
    assert skel.joints["neck_c7"].z > 0.0
    assert skel.trunk_angle == 45.0


def test_generate_3d_skeleton_plotly_figure():
    """Test generating interactive 3D Plotly figure."""
    skel = lift_2d_to_3d_skeleton(np.empty((0, 3)), trunk_flexion_deg=30.0)
    stress = compute_l5s1_spinal_compression(30.0)

    fig = generate_3d_skeleton_plotly_figure(skel, spine_stress=stress, height=350)
    assert fig is not None
    # Verify traces are added: Limb Segments, Vertebral Column, L5/S1 Disc, Anatomical Joints, COG
    trace_names = [trace.name for trace in fig.data]
    assert "Limb Segments" in trace_names
    assert "Vertebral Column" in trace_names
    assert "L5/S1 Disc Joint" in trace_names
    assert "Center of Gravity (COG)" in trace_names
