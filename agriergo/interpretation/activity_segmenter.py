"""
Activity Segmenter — Merges frame-level posture labels into activity bouts.

Groups consecutive frames with the same posture label into continuous
activity bouts, filters out noise, and classifies bouts as "work" or "rest".
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    MIN_BOUT_DURATION,
    REST_STILLNESS_THRESHOLD,
    REST_MIN_DURATION,
)
from agriergo.interpretation.posture_classifier import PostureLabel


@dataclass
class ActivityBout:
    """A continuous period of a single activity/posture."""
    worker_id: int
    activity: PostureLabel
    start_time: float              # Seconds from video start
    end_time: float                # Seconds from video start
    duration: float                # Seconds
    start_frame: int
    end_frame: int
    is_rest: bool = False          # Whether this bout is classified as rest
    avg_displacement: float = 0.0  # Average keypoint displacement during bout


@dataclass
class FrameRecord:
    """Per-frame data for a single worker."""
    frame_idx: int
    timestamp: float
    posture: PostureLabel
    displacement: Optional[float] = None


class ActivitySegmenter:
    """
    Segments frame-level posture labels into continuous activity bouts.

    Process:
    1. Group consecutive frames with the same posture label.
    2. Filter out bouts shorter than MIN_BOUT_DURATION.
    3. Merge adjacent bouts of the same type (after filtering).
    4. Classify bouts as "work" or "rest" based on movement level.
    """

    def __init__(
        self,
        min_bout_duration: float = MIN_BOUT_DURATION,
        rest_stillness_threshold: float = REST_STILLNESS_THRESHOLD,
        rest_min_duration: float = REST_MIN_DURATION,
    ):
        self.min_bout_duration = min_bout_duration
        self.rest_stillness_threshold = rest_stillness_threshold
        self.rest_min_duration = rest_min_duration

    def segment(
        self,
        worker_id: int,
        frame_records: List[FrameRecord],
    ) -> List[ActivityBout]:
        """
        Segment frame records into activity bouts.

        Args:
            worker_id: Worker identifier.
            frame_records: Chronologically ordered list of per-frame records.

        Returns:
            List of ActivityBout objects.
        """
        if not frame_records:
            return []

        # Step 1: Group consecutive identical labels
        raw_bouts = self._group_consecutive(worker_id, frame_records)

        # Step 2: Filter short bouts (absorb into neighbors)
        filtered_bouts = self._filter_short_bouts(raw_bouts)

        # Step 3: Merge adjacent same-label bouts
        merged_bouts = self._merge_adjacent(filtered_bouts)

        # Step 4: Classify work vs rest
        classified_bouts = self._classify_rest(merged_bouts, frame_records)

        return classified_bouts

    def _group_consecutive(
        self, worker_id: int, records: List[FrameRecord]
    ) -> List[ActivityBout]:
        """Group consecutive frames with the same posture into bouts."""
        bouts = []
        current_label = records[0].posture
        start_idx = 0
        displacements = []

        for i, record in enumerate(records):
            if record.displacement is not None:
                displacements.append(record.displacement)

            if record.posture != current_label or i == len(records) - 1:
                # End of current bout
                end_idx = i if record.posture != current_label else i
                if record.posture == current_label:
                    end_idx = i

                start_time = records[start_idx].timestamp
                end_time = records[end_idx].timestamp

                bout = ActivityBout(
                    worker_id=worker_id,
                    activity=current_label,
                    start_time=start_time,
                    end_time=end_time,
                    duration=round(end_time - start_time, 2),
                    start_frame=records[start_idx].frame_idx,
                    end_frame=records[end_idx].frame_idx,
                    avg_displacement=float(np.mean(displacements)) if displacements else 0.0,
                )
                bouts.append(bout)

                # Start new bout
                if record.posture != current_label:
                    current_label = record.posture
                    start_idx = i
                    displacements = [record.displacement] if record.displacement is not None else []

        return bouts

    def _filter_short_bouts(self, bouts: List[ActivityBout]) -> List[ActivityBout]:
        """Remove bouts shorter than minimum duration, absorbing into neighbors."""
        if len(bouts) <= 1:
            return bouts

        filtered = []
        for bout in bouts:
            if bout.duration >= self.min_bout_duration:
                filtered.append(bout)
            elif filtered:
                # Absorb into previous bout by extending its end time
                filtered[-1].end_time = bout.end_time
                filtered[-1].end_frame = bout.end_frame
                filtered[-1].duration = round(
                    filtered[-1].end_time - filtered[-1].start_time, 2
                )

        return filtered if filtered else bouts[:1]

    def _merge_adjacent(self, bouts: List[ActivityBout]) -> List[ActivityBout]:
        """Merge adjacent bouts with the same activity label."""
        if len(bouts) <= 1:
            return bouts

        merged = [bouts[0]]
        for bout in bouts[1:]:
            if bout.activity == merged[-1].activity:
                # Merge into previous
                merged[-1].end_time = bout.end_time
                merged[-1].end_frame = bout.end_frame
                merged[-1].duration = round(
                    merged[-1].end_time - merged[-1].start_time, 2
                )
                merged[-1].avg_displacement = (
                    merged[-1].avg_displacement + bout.avg_displacement
                ) / 2
            else:
                merged.append(bout)

        return merged

    def _classify_rest(
        self,
        bouts: List[ActivityBout],
        frame_records: List[FrameRecord],
    ) -> List[ActivityBout]:
        """
        Classify bouts as work or rest based on movement level.

        Rest criteria:
        - Posture is STANDING or SITTING
        - Average displacement below stillness threshold
        - Duration >= minimum rest duration
        """
        for bout in bouts:
            if bout.activity in (PostureLabel.STANDING, PostureLabel.SITTING):
                if (bout.avg_displacement < self.rest_stillness_threshold and
                        bout.duration >= self.rest_min_duration):
                    bout.is_rest = True

        return bouts
