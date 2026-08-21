"""
Video Processor — Frame extraction, sampling, and video I/O.

Handles opening video files, extracting metadata, and yielding sampled
frames at a configurable effective FPS for downstream processing.
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Generator, Tuple, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    FRAME_SAMPLE_FPS, SUPPORTED_FORMATS, get_adaptive_fps
)


@dataclass
class VideoMetadata:
    """Metadata extracted from a video file."""
    filepath: str
    filename: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float
    codec: str


class VideoProcessor:
    """
    Handles video file I/O, metadata extraction, and frame sampling.

    Usage:
        vp = VideoProcessor("path/to/video.mp4")
        print(vp.metadata)
        for frame_idx, timestamp, frame in vp.sample_frames(fps=5):
            # process frame ...
    """

    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if self.video_path.suffix.lower() not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format '{self.video_path.suffix}'. "
                f"Supported: {SUPPORTED_FORMATS}"
            )
        self._cap: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[VideoMetadata] = None

    @property
    def metadata(self) -> VideoMetadata:
        """Extract and cache video metadata."""
        if self._metadata is None:
            cap = cv2.VideoCapture(str(self.video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {self.video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

            duration = total_frames / fps if fps > 0 else 0.0

            self._metadata = VideoMetadata(
                filepath=str(self.video_path),
                filename=self.video_path.name,
                width=width,
                height=height,
                fps=fps,
                total_frames=total_frames,
                duration_seconds=round(duration, 2),
                codec=codec,
            )
            cap.release()

        return self._metadata



    def sample_frames(
        self, fps: Optional[float] = None, speed_mode: str = "Balanced Fast"
    ) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """
        Yield frames sampled at the specified or adaptive effective FPS.
        Uses robust sequential stream decoding (grab skipping) to prevent HEVC/H.265
        and MPEG-TS PPS/POC seek corruption errors.

        Args:
            fps: Target sampling rate (frames per second). If None, uses get_adaptive_fps().
            speed_mode: "Lightning Fast", "Balanced Fast", or "High Precision Research".

        Yields:
            Tuple of (frame_index, timestamp_seconds, frame_bgr_array)
        """
        meta = self.metadata
        effective_fps = fps if fps is not None else get_adaptive_fps(meta.duration_seconds, speed_mode=speed_mode)

        if effective_fps >= meta.fps:
            frame_interval = 1
        else:
            frame_interval = max(1, int(round(meta.fps / effective_fps)))

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        frame_idx = 0
        consecutive_failures = 0
        max_failures = 30

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures or (meta.total_frames > 0 and frame_idx >= meta.total_frames):
                        break
                    frame_idx += 1
                    continue

                consecutive_failures = 0
                # Robust timestamp calculation
                pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                if pos_msec > 0:
                    timestamp = pos_msec / 1000.0
                else:
                    timestamp = frame_idx / meta.fps if meta.fps > 0 else 0.0

                yield frame_idx, round(timestamp, 3), frame
                frame_idx += 1

                # Fast skip the next (frame_interval - 1) frames using cap.grab()
                # This maintains valid HEVC/H.265 GOP reference chains without PPS/POC errors
                if frame_interval > 1:
                    for _ in range(frame_interval - 1):
                        grabbed = cap.grab()
                        frame_idx += 1
                        if not grabbed:
                            break
        finally:
            cap.release()

    def get_frame_at(self, timestamp: float) -> Optional[np.ndarray]:
        """
        Seek to a specific timestamp and return the frame.

        Args:
            timestamp: Time in seconds.

        Returns:
            BGR frame array, or None if seek fails.
        """
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            return None

        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = cap.read()
        cap.release()

        return frame if ret else None

    def get_total_sampled_frames(self, fps: float = FRAME_SAMPLE_FPS) -> int:
        """Estimate the total number of frames that will be sampled."""
        meta = self.metadata
        if fps >= meta.fps:
            return meta.total_frames
        frame_interval = int(round(meta.fps / fps))
        return meta.total_frames // frame_interval
