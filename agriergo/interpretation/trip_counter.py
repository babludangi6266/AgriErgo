"""
Trip Counter — Detects back-and-forth worker movements (trips).

A "trip" is defined as a worker moving significantly in one direction,
then reversing direction (e.g., walking from the field to a collection
point and back). Detection uses trajectory direction-reversal analysis.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    TRIP_DIRECTION_CHANGE_ANGLE,
    TRIP_MIN_DISPLACEMENT,
    TRIP_SUSTAIN_FRAMES,
    KP_LEFT_HIP, KP_RIGHT_HIP,
)


@dataclass
class Trip:
    """A single detected trip (one direction of travel)."""
    start_time: float
    end_time: float
    start_position: Tuple[float, float]
    end_position: Tuple[float, float]
    distance_pixels: float
    direction_degrees: float


@dataclass
class TripCountResult:
    """Result of trip detection for a worker."""
    trip_count: int
    trips: List[Trip]
    total_distance_pixels: float
    trajectory: List[Tuple[float, float, float]]  # (time, x, y)


class TripCounter:
    """
    Counts worker trips by detecting direction reversals in their trajectory.

    A trip reversal is detected when:
    1. The worker has traveled at least TRIP_MIN_DISPLACEMENT pixels
    2. The direction changes by more than TRIP_DIRECTION_CHANGE_ANGLE degrees
    3. The new direction is sustained for TRIP_SUSTAIN_FRAMES frames
    """

    def __init__(
        self,
        direction_change_angle: float = TRIP_DIRECTION_CHANGE_ANGLE,
        min_displacement: float = TRIP_MIN_DISPLACEMENT,
        sustain_frames: int = TRIP_SUSTAIN_FRAMES,
    ):
        self.direction_change_angle = direction_change_angle
        self.min_displacement = min_displacement
        self.sustain_frames = sustain_frames

    def count_trips(
        self,
        trajectory: List[Tuple[float, float, float]],
    ) -> TripCountResult:
        """
        Count trips from a worker's trajectory.

        Args:
            trajectory: List of (timestamp, x, y) tuples representing
                        the worker's centroid position over time.

        Returns:
            TripCountResult with trip count and details.
        """
        if len(trajectory) < self.sustain_frames * 2:
            return TripCountResult(
                trip_count=0,
                trips=[],
                total_distance_pixels=0.0,
                trajectory=trajectory,
            )

        # Smooth trajectory to reduce noise
        smoothed = self._smooth_trajectory(trajectory)

        # Detect direction changes
        trips = self._detect_reversals(smoothed)

        # Calculate total distance
        total_dist = self._total_distance(smoothed)

        return TripCountResult(
            trip_count=len(trips),
            trips=trips,
            total_distance_pixels=round(total_dist, 1),
            trajectory=trajectory,
        )

    def _smooth_trajectory(
        self, trajectory: List[Tuple[float, float, float]], window: int = 5
    ) -> List[Tuple[float, float, float]]:
        """Apply moving average smoothing to trajectory coordinates."""
        if len(trajectory) < window:
            return trajectory

        times = [t[0] for t in trajectory]
        xs = np.array([t[1] for t in trajectory])
        ys = np.array([t[2] for t in trajectory])

        # Simple moving average
        kernel = np.ones(window) / window
        xs_smooth = np.convolve(xs, kernel, mode='same')
        ys_smooth = np.convolve(ys, kernel, mode='same')

        return list(zip(times, xs_smooth.tolist(), ys_smooth.tolist()))

    def _detect_reversals(
        self, trajectory: List[Tuple[float, float, float]]
    ) -> List[Trip]:
        """Detect direction reversals in the smoothed trajectory."""
        trips = []
        n = len(trajectory)

        if n < 3:
            return trips

        # Compute direction at each point (angle in degrees)
        directions = []
        for i in range(1, n):
            dx = trajectory[i][1] - trajectory[i - 1][1]
            dy = trajectory[i][2] - trajectory[i - 1][2]
            angle = np.degrees(np.arctan2(dy, dx))
            directions.append(angle)

        # Track trip segments
        trip_start_idx = 0
        trip_start_dir = directions[0] if directions else 0.0
        accumulated_displacement = 0.0
        sustained_count = 0
        new_direction = None

        for i in range(1, len(directions)):
            # Accumulate displacement from trip start
            dx = trajectory[i + 1][1] - trajectory[trip_start_idx][1]
            dy = trajectory[i + 1][2] - trajectory[trip_start_idx][2]
            accumulated_displacement = np.sqrt(dx**2 + dy**2)

            # Check for direction change
            angle_diff = self._angle_difference(directions[i], trip_start_dir)

            if (abs(angle_diff) > self.direction_change_angle and
                    accumulated_displacement > self.min_displacement):
                if new_direction is None:
                    new_direction = directions[i]
                    sustained_count = 1
                else:
                    # Check if new direction is sustained
                    if abs(self._angle_difference(directions[i], new_direction)) < 45:
                        sustained_count += 1
                    else:
                        sustained_count = 0
                        new_direction = None

                if sustained_count >= self.sustain_frames:
                    # Trip reversal confirmed
                    trip = Trip(
                        start_time=trajectory[trip_start_idx][0],
                        end_time=trajectory[i + 1][0],
                        start_position=(
                            trajectory[trip_start_idx][1],
                            trajectory[trip_start_idx][2],
                        ),
                        end_position=(
                            trajectory[i + 1][1],
                            trajectory[i + 1][2],
                        ),
                        distance_pixels=round(accumulated_displacement, 1),
                        direction_degrees=round(trip_start_dir, 1),
                    )
                    trips.append(trip)

                    # Reset for next trip
                    trip_start_idx = i + 1
                    trip_start_dir = directions[i]
                    accumulated_displacement = 0.0
                    sustained_count = 0
                    new_direction = None
            else:
                sustained_count = 0
                new_direction = None

        return trips

    @staticmethod
    def _angle_difference(angle1: float, angle2: float) -> float:
        """Compute signed angular difference, normalized to [-180, 180]."""
        diff = angle1 - angle2
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        return diff

    @staticmethod
    def _total_distance(trajectory: List[Tuple[float, float, float]]) -> float:
        """Compute total path distance in pixels."""
        total = 0.0
        for i in range(1, len(trajectory)):
            dx = trajectory[i][1] - trajectory[i - 1][1]
            dy = trajectory[i][2] - trajectory[i - 1][2]
            total += np.sqrt(dx**2 + dy**2)
        return total

    @staticmethod
    def extract_centroid(
        keypoints: np.ndarray, confidences: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        """
        Extract worker centroid from keypoints (mid-hip position).

        Args:
            keypoints: Shape (17, 2).
            confidences: Shape (17,).

        Returns:
            (x, y) centroid or None if hip keypoints are not confident.
        """
        min_conf = 0.3
        l_valid = confidences[KP_LEFT_HIP] >= min_conf
        r_valid = confidences[KP_RIGHT_HIP] >= min_conf

        if l_valid and r_valid:
            mid = (keypoints[KP_LEFT_HIP] + keypoints[KP_RIGHT_HIP]) / 2
            return (float(mid[0]), float(mid[1]))
        elif l_valid:
            return (float(keypoints[KP_LEFT_HIP][0]), float(keypoints[KP_LEFT_HIP][1]))
        elif r_valid:
            return (float(keypoints[KP_RIGHT_HIP][0]), float(keypoints[KP_RIGHT_HIP][1]))
        return None
