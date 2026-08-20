"""
Unit tests for Spatial-Temporal Tracklet Stitcher.
"""

import pytest
from agriergo.perception.tracklet_stitcher import TrackletStitcher


def test_tracklet_stitcher_merges_fragmented_ids():
    """Test that two non-overlapping tracklets in the same spatial vicinity are merged."""
    stitcher = TrackletStitcher(max_time_gap_seconds=60.0, max_spatial_distance_pixels=100.0)

    # Worker A (Raw ID 100): t=0s to 30s at (100, 200) -> (120, 210)
    traj_100 = [
        (0.0, 100.0, 200.0),
        (10.0, 105.0, 205.0),
        (20.0, 110.0, 208.0),
        (30.0, 120.0, 210.0),
    ]

    # Worker A fragmented (Raw ID 200): t=35s to 70s starting nearby at (125, 215)
    traj_200 = [
        (35.0, 125.0, 215.0),
        (45.0, 130.0, 220.0),
        (60.0, 140.0, 230.0),
        (70.0, 150.0, 240.0),
    ]

    # Worker B (Raw ID 300): t=0s to 60s in a completely different row at (600, 800)
    traj_300 = [
        (0.0, 600.0, 800.0),
        (20.0, 610.0, 810.0),
        (40.0, 620.0, 820.0),
        (60.0, 630.0, 830.0),
    ]

    trajectories = {
        100: traj_100,
        200: traj_200,
        300: traj_300,
    }

    raw_to_unified, peak_concurrency = stitcher.stitch_tracklets(trajectories)

    # Raw IDs 100 and 200 should map to the SAME unified ID
    assert raw_to_unified[100] == raw_to_unified[200]

    # Raw ID 300 should map to a DIFFERENT unified ID
    assert raw_to_unified[100] != raw_to_unified[300]

    # Total unified physical workers should be 2
    unique_unified = set(raw_to_unified.values())
    assert len(unique_unified) == 2

    # Peak concurrency is 2 (at t=0..30, Worker A and Worker B are both present)
    assert peak_concurrency == 2


def test_tracklet_stitcher_does_not_merge_simultaneous_workers():
    """Test that two workers active at the same time are NOT merged even if nearby."""
    stitcher = TrackletStitcher(max_time_gap_seconds=60.0, max_spatial_distance_pixels=500.0)

    # Worker 1: t=10s to 50s
    traj_1 = [
        (10.0, 100.0, 100.0),
        (20.0, 105.0, 105.0),
        (30.0, 110.0, 110.0),
        (50.0, 120.0, 120.0),
    ]

    # Worker 2: also active t=15s to 55s (simultaneous overlap)
    traj_2 = [
        (15.0, 110.0, 110.0),
        (25.0, 115.0, 115.0),
        (35.0, 120.0, 120.0),
        (55.0, 130.0, 130.0),
    ]

    trajectories = {1: traj_1, 2: traj_2}
    raw_to_unified, peak_concurrency = stitcher.stitch_tracklets(trajectories)

    # Must NOT merge because they are active simultaneously
    assert raw_to_unified[1] != raw_to_unified[2]
    assert peak_concurrency == 2
