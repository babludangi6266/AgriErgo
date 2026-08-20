"""
Spatial-Temporal Tracklet Stitcher — Merges fragmented ByteTrack IDs into true physical workers.

In long agricultural videos, workers stoop, turn, or get temporarily occluded by crops,
causing trackers to drop IDs and re-assign new IDs (ID fragmentation).
This module uses spatial proximity, temporal non-overlap, and trajectory continuity
to stitch fragmented tracklets back into unified physical worker identities.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Optional, Any
from collections import defaultdict


@dataclass
class TrackletMeta:
    """Metadata summary of a single tracked worker trajectory."""
    raw_id: int
    start_time: float
    end_time: float
    duration: float
    total_frames: int
    start_pos: Tuple[float, float]
    end_pos: Tuple[float, float]
    mean_pos: Tuple[float, float]


class TrackletStitcher:
    """
    Stitches fragmented tracklets across time using spatial-temporal proximity.
    """

    def __init__(
        self,
        max_time_gap_seconds: float = 120.0,
        max_spatial_distance_pixels: float = 400.0,
        min_tracklet_frames: int = 3,
    ):
        """
        Args:
            max_time_gap_seconds: Maximum time gap between tracklets to allow merging (default: 120s).
            max_spatial_distance_pixels: Maximum allowable pixel distance between tracklet end and start.
            min_tracklet_frames: Minimum frame count for a valid tracklet.
        """
        self.max_time_gap = max_time_gap_seconds
        self.max_spatial_distance = max_spatial_distance_pixels
        self.min_tracklet_frames = min_tracklet_frames

    def compute_peak_concurrency(
        self,
        worker_trajectories: Dict[int, List[Tuple[float, float, float]]]
    ) -> int:
        """
        Compute the peak number of simultaneous physical workers visible across time intervals.
        """
        meta_dict = self.extract_metadata(worker_trajectories)
        if not meta_dict:
            return 0

        # Event-based interval sweep
        events = []
        for meta in meta_dict.values():
            events.append((meta.start_time, 1))
            events.append((meta.end_time, -1))

        # Sort events by time; if times are equal, start (+1) comes before end (-1)
        events.sort(key=lambda e: (e[0], -e[1]))

        current_active = 0
        peak = 0
        for _, change in events:
            current_active += change
            if current_active > peak:
                peak = current_active

        return max(1, peak)

    def extract_metadata(
        self,
        worker_trajectories: Dict[int, List[Tuple[float, float, float]]]
    ) -> Dict[int, TrackletMeta]:
        """Extract spatial and temporal boundaries for each raw worker ID."""
        metadata = {}
        for wid, traj in worker_trajectories.items():
            if len(traj) < self.min_tracklet_frames:
                continue

            timestamps = [t[0] for t in traj]
            xs = [t[1] for t in traj]
            ys = [t[2] for t in traj]

            start_t = min(timestamps)
            end_t = max(timestamps)
            start_p = (xs[0], ys[0])
            end_p = (xs[-1], ys[-1])
            mean_p = (float(np.mean(xs)), float(np.mean(ys)))

            metadata[wid] = TrackletMeta(
                raw_id=wid,
                start_time=start_t,
                end_time=end_t,
                duration=max(1.0, end_t - start_t),
                total_frames=len(traj),
                start_pos=start_p,
                end_pos=end_p,
                mean_pos=mean_p,
            )
        return metadata

    def stitch_tracklets(
        self,
        worker_trajectories: Dict[int, List[Tuple[float, float, float]]]
    ) -> Tuple[Dict[int, int], int]:
        """
        Merge fragmented tracklets and return (mapping_dict, peak_concurrency).

        Returns:
            mapping: dict mapping raw_id -> unified_sequential_id (1, 2, 3...)
            peak_concurrency: maximum simultaneous workers seen at once.
        """
        meta_dict = self.extract_metadata(worker_trajectories)
        peak_concurrency = self.compute_peak_concurrency(worker_trajectories)

        if not meta_dict:
            return {}, 0

        # Sort tracklets chronologically by start time
        sorted_tracklets = sorted(meta_dict.values(), key=lambda t: t.start_time)

        # Disjoint Set Union (Union-Find) for merging
        parent: Dict[int, int] = {t.raw_id: t.raw_id for t in sorted_tracklets}

        def find(i: int) -> int:
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]

        def union(i: int, j: int):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_j] = root_i

        # Greedy forward temporal matching
        merged_targets: Set[int] = set()

        for i, t1 in enumerate(sorted_tracklets):
            best_match: Optional[int] = None
            best_distance = float('inf')

            for j in range(i + 1, len(sorted_tracklets)):
                t2 = sorted_tracklets[j]

                # If t2 is already merged to another prior tracklet, skip
                if t2.raw_id in merged_targets:
                    continue

                # Ensure non-overlapping in time (allow at most 2s buffer for tracker transition)
                time_gap = t2.start_time - t1.end_time
                if time_gap < -2.0:
                    continue  # Significant temporal overlap -> cannot be the same physical person

                if time_gap > self.max_time_gap:
                    continue  # Time gap too long

                # Calculate spatial Euclidean distance between t1's end position and t2's start position
                spatial_dist = np.sqrt(
                    (t2.start_pos[0] - t1.end_pos[0]) ** 2 +
                    (t2.start_pos[1] - t1.end_pos[1]) ** 2
                )

                # Also consider mean spatial cluster distance
                cluster_dist = np.sqrt(
                    (t2.mean_pos[0] - t1.mean_pos[0]) ** 2 +
                    (t2.mean_pos[1] - t1.mean_pos[1]) ** 2
                )

                effective_dist = min(spatial_dist, cluster_dist)

                if effective_dist <= self.max_spatial_distance and effective_dist < best_distance:
                    best_distance = effective_dist
                    best_match = t2.raw_id

            if best_match is not None:
                union(t1.raw_id, best_match)
                merged_targets.add(best_match)

        # Group raw IDs by unified cluster root
        clusters: Dict[int, List[int]] = defaultdict(list)
        for t in sorted_tracklets:
            root = find(t.raw_id)
            clusters[root].append(t.raw_id)

        # Sort clusters by total presence duration (descending)
        def cluster_total_frames(cluster_ids: List[int]) -> int:
            return sum(meta_dict[wid].total_frames for wid in cluster_ids)

        sorted_clusters = sorted(
            clusters.values(),
            key=cluster_total_frames,
            reverse=True
        )

        # Assign clean human-readable sequential IDs (1, 2, 3...)
        raw_to_unified: Dict[int, int] = {}
        for sequential_id, cluster_ids in enumerate(sorted_clusters, start=1):
            for raw_id in cluster_ids:
                raw_to_unified[raw_id] = sequential_id

        return raw_to_unified, peak_concurrency
