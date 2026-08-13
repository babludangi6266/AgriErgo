"""
Repetition Detector — Detects periodic/repetitive motions from joint angle time series.

Analyzes the oscillation of specific joint angles (e.g., elbow flexion during
weeding strokes, wrist movement during harvesting) to estimate repetition
frequency in cycles per minute.
"""

import numpy as np
from scipy import signal
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    REPETITION_FREQ_MIN,
    REPETITION_FREQ_MAX,
    REPETITION_PEAK_PROMINENCE,
)


@dataclass
class RepetitionResult:
    """Result of repetitive motion analysis for joint angle series."""
    frequency_hz: Optional[float]       # Dominant frequency in Hz
    cycles_per_minute: Optional[float]  # Frequency in cycles/min
    peak_count: int                     # Number of peaks detected
    confidence: float                   # Detection confidence (0-1)
    is_repetitive: bool                 # Whether motion is classified as repetitive
    primary_joint: str = "elbow"        # Joint channel exhibiting highest spectral periodicity


class RepetitionDetector:
    """
    Detects periodic/repetitive motions across multiple joint channels.

    Evaluates time series from:
    - Elbow angle (weeding, pruning strokes)
    - Wrist angle (harvesting, picking)
    - Shoulder elevation (hoeing, digging)
    - Trunk flexion (stooped hoeing)

    Selects the joint displaying the highest periodicity confidence.
    """

    def __init__(
        self,
        freq_min: float = REPETITION_FREQ_MIN,
        freq_max: float = REPETITION_FREQ_MAX,
        peak_prominence: float = REPETITION_PEAK_PROMINENCE,
    ):
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.peak_prominence = peak_prominence

    def detect_multi_joint(
        self,
        joint_series_dict: Dict[str, np.ndarray],
        sample_fps: float,
        min_duration_seconds: float = 5.0,
    ) -> RepetitionResult:
        """
        Analyze multiple joint angle channels and return the dominant repetition result.

        Args:
            joint_series_dict: Dictionary mapping joint_name -> 1D angle series.
                               e.g., {"elbow": ..., "wrist": ..., "shoulder": ..., "trunk": ...}
            sample_fps: Sampling rate in frames per second.
            min_duration_seconds: Minimum series duration.

        Returns:
            RepetitionResult for the joint with highest repetition confidence.
        """
        best_result = RepetitionResult(
            frequency_hz=None,
            cycles_per_minute=None,
            peak_count=0,
            confidence=0.0,
            is_repetitive=False,
            primary_joint="elbow",
        )

        for joint_name, series in joint_series_dict.items():
            if series is None or len(series) == 0:
                continue

            res = self.detect_frequency(
                series, sample_fps, min_duration_seconds=min_duration_seconds
            )
            res.primary_joint = joint_name

            if res.confidence > best_result.confidence:
                best_result = res

        return best_result

    def detect_frequency(
        self,
        angle_series: np.ndarray,
        sample_fps: float,
        min_duration_seconds: float = 5.0,
    ) -> RepetitionResult:
        """
        Analyze a time series of joint angles for repetitive motion.

        Args:
            angle_series: 1D array of joint angle values over time.
            sample_fps: Sampling rate in frames per second.
            min_duration_seconds: Minimum series duration to attempt detection.

        Returns:
            RepetitionResult with frequency and confidence.
        """
        # Validate input
        if len(angle_series) < int(min_duration_seconds * sample_fps):
            return RepetitionResult(
                frequency_hz=None,
                cycles_per_minute=None,
                peak_count=0,
                confidence=0.0,
                is_repetitive=False,
            )

        # Remove NaN values and interpolate gaps
        clean_series = self._clean_series(angle_series)
        if len(clean_series) < 10:
            return RepetitionResult(
                frequency_hz=None,
                cycles_per_minute=None,
                peak_count=0,
                confidence=0.0,
                is_repetitive=False,
            )

        # Detrend the signal (remove slow drift)
        detrended = signal.detrend(clean_series)

        # Method 1: Peak detection in time domain
        time_result = self._detect_from_peaks(detrended, sample_fps)

        # Method 2: FFT-based spectral analysis
        fft_result = self._detect_from_fft(detrended, sample_fps)

        # Combine results — prefer FFT if both agree, otherwise use higher confidence
        return self._combine_results(time_result, fft_result)

    def _clean_series(self, series: np.ndarray) -> np.ndarray:
        """Remove NaNs and interpolate gaps."""
        if np.all(np.isnan(series)):
            return np.array([])

        # Replace NaN with interpolated values
        mask = np.isnan(series)
        if np.any(mask):
            indices = np.arange(len(series))
            valid = ~mask
            if np.sum(valid) < 3:
                return np.array([])
            series = np.interp(indices, indices[valid], series[valid])

        return series

    def _detect_from_peaks(
        self, series: np.ndarray, fps: float
    ) -> RepetitionResult:
        """Detect repetition frequency via peak detection in time domain."""
        # Find peaks with minimum prominence
        peaks, properties = signal.find_peaks(
            series,
            prominence=self.peak_prominence,
            distance=int(fps / self.freq_max) if self.freq_max > 0 else 1,
        )

        if len(peaks) < 3:
            return RepetitionResult(
                frequency_hz=None,
                cycles_per_minute=None,
                peak_count=len(peaks),
                confidence=0.0,
                is_repetitive=False,
            )

        # Calculate inter-peak intervals
        intervals = np.diff(peaks) / fps  # in seconds
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)

        if mean_interval <= 0:
            return RepetitionResult(
                frequency_hz=None,
                cycles_per_minute=None,
                peak_count=len(peaks),
                confidence=0.0,
                is_repetitive=False,
            )

        freq_hz = 1.0 / mean_interval

        # Check if frequency is in expected range
        if not (self.freq_min <= freq_hz <= self.freq_max):
            return RepetitionResult(
                frequency_hz=freq_hz,
                cycles_per_minute=freq_hz * 60,
                peak_count=len(peaks),
                confidence=0.2,
                is_repetitive=False,
            )

        # Confidence based on regularity of intervals
        cv = std_interval / mean_interval if mean_interval > 0 else 1.0
        confidence = max(0.0, min(1.0, 1.0 - cv))

        return RepetitionResult(
            frequency_hz=round(freq_hz, 3),
            cycles_per_minute=round(freq_hz * 60, 1),
            peak_count=len(peaks),
            confidence=round(confidence, 2),
            is_repetitive=confidence > 0.4,
        )

    def _detect_from_fft(
        self, series: np.ndarray, fps: float
    ) -> RepetitionResult:
        """Detect dominant frequency via FFT spectral analysis."""
        n = len(series)
        if n < 16:
            return RepetitionResult(
                frequency_hz=None,
                cycles_per_minute=None,
                peak_count=0,
                confidence=0.0,
                is_repetitive=False,
            )

        # Apply Hanning window to reduce spectral leakage
        windowed = series * np.hanning(n)

        # Compute FFT
        fft_vals = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(n, d=1.0 / fps)

        # Filter to expected frequency range
        mask = (freqs >= self.freq_min) & (freqs <= self.freq_max)
        if not np.any(mask):
            return RepetitionResult(
                frequency_hz=None,
                cycles_per_minute=None,
                peak_count=0,
                confidence=0.0,
                is_repetitive=False,
            )

        filtered_fft = fft_vals[mask]
        filtered_freqs = freqs[mask]

        # Find dominant frequency
        peak_idx = np.argmax(filtered_fft)
        dominant_freq = filtered_freqs[peak_idx]
        peak_power = filtered_fft[peak_idx]

        # Confidence: ratio of peak to mean power
        mean_power = np.mean(filtered_fft)
        confidence = min(1.0, (peak_power / (mean_power + 1e-8) - 1) / 5)
        confidence = max(0.0, confidence)

        return RepetitionResult(
            frequency_hz=round(float(dominant_freq), 3),
            cycles_per_minute=round(float(dominant_freq * 60), 1),
            peak_count=0,  # Not applicable for FFT
            confidence=round(float(confidence), 2),
            is_repetitive=confidence > 0.3,
        )

    def _combine_results(
        self, time_result: RepetitionResult, fft_result: RepetitionResult
    ) -> RepetitionResult:
        """Combine time-domain and frequency-domain results."""
        # If both detect repetition and agree on frequency, boost confidence
        if time_result.is_repetitive and fft_result.is_repetitive:
            if (time_result.frequency_hz is not None and
                    fft_result.frequency_hz is not None):
                freq_diff = abs(time_result.frequency_hz - fft_result.frequency_hz)
                if freq_diff < 0.2:  # Frequencies agree within 0.2 Hz
                    return RepetitionResult(
                        frequency_hz=round(
                            (time_result.frequency_hz + fft_result.frequency_hz) / 2, 3
                        ),
                        cycles_per_minute=round(
                            (time_result.cycles_per_minute + fft_result.cycles_per_minute) / 2, 1
                        ),
                        peak_count=time_result.peak_count,
                        confidence=min(1.0, max(
                            time_result.confidence, fft_result.confidence
                        ) + 0.15),
                        is_repetitive=True,
                    )

        # Return whichever has higher confidence
        if time_result.confidence >= fft_result.confidence:
            return time_result
        return fft_result
