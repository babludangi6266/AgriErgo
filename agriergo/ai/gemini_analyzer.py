"""
Gemini Multimodal Analyzer — AI-Powered Agricultural Ergonomic Assessment.

Uses Google Gemini 2.0 Flash to analyze sampled video keyframes alongside
biomechanical data (joint angles, posture distributions, REBA/RULA scores)
and produce:
  1. Context-aware agricultural task classification
  2. Expert-level ergonomic risk narratives
  3. Worker-specific actionable recommendations
  4. Scene context (crop type, terrain, weather, tools)

Gracefully degrades to rule-based classification when no API key is available.
"""

import os
import io
import json
import hashlib
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np
import cv2

logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    GEMINI_MODEL,
    GEMINI_MAX_FRAMES_PER_WORKER,
    GEMINI_FRAME_RESIZE,
    GEMINI_TEMPERATURE,
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_API_KEY_ENV,
)


@dataclass
class AIAnalysisResult:
    """Structured output from Gemini multimodal ergonomic analysis."""
    ai_task_classification: str = "Unknown"
    ai_task_confidence: float = 0.0
    ai_crop_detected: str = "Unknown"
    ai_terrain_type: str = "Unknown"
    ai_risk_narrative: str = ""
    ai_recommendations: List[str] = field(default_factory=list)
    ai_tools_identified: List[str] = field(default_factory=list)
    ai_severity_rating: str = "Unknown"
    ai_working_conditions: str = ""
    ai_model_used: str = ""
    ai_analysis_success: bool = False


# ─────────────────────────────────────────────────────
# Prompt Engineering
# ─────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert agricultural ergonomics researcher with deep knowledge of:
- Occupational biomechanics and musculoskeletal disorder (MSD) risk assessment
- Agricultural field work activities across diverse crops (rice, wheat, turmeric, vegetables, fruits, sugarcane, etc.)
- Validated ergonomic assessment tools: REBA, RULA, NIOSH Lifting Equation, ISO 11226
- Work physiology: fatigue, drudgery, and physical strain in manual farming

You are analyzing video frames from an agricultural field along with computed biomechanical measurements.
Your task is to provide an expert ergonomic assessment based on the visual evidence and numerical data."""


def _build_analysis_prompt(biomechanical_context: Dict[str, Any]) -> str:
    """Build the structured analysis prompt with biomechanical context."""
    ctx = biomechanical_context
    return f"""Analyze the provided video frames of an agricultural field worker and the biomechanical measurements below.

## Biomechanical Measurements (Computed from Pose Estimation)
- **Tracked Duration**: {ctx.get('tracked_time_formatted', 'N/A')}
- **Dominant Posture**: {ctx.get('dominant_posture', 'N/A')}
- **Posture Distribution**: {json.dumps(ctx.get('posture_distribution', {}), indent=2)}
- **Average Trunk Flexion**: {ctx.get('avg_trunk_flexion', 'N/A')}°
- **Max Trunk Flexion**: {ctx.get('max_trunk_flexion', 'N/A')}°
- **Average Knee Angle**: {ctx.get('avg_knee_angle', 'N/A')}°
- **REBA Score**: {ctx.get('reba_score', 'N/A')} ({ctx.get('reba_risk', 'N/A')})
- **RULA Score**: {ctx.get('rula_score', 'N/A')} ({ctx.get('rula_action', 'N/A')})
- **Drudgery Index**: {ctx.get('drudgery_index', 'N/A')}/100 ({ctx.get('drudgery_category', 'N/A')})
- **Repetitive Motion**: {ctx.get('cycles_per_minute', 'N/A')} cycles/min (Joint: {ctx.get('repetitive_joint', 'N/A')})
- **Shoulder Elevation > 45°**: {ctx.get('shoulder_above_45_pct', 'N/A')}%
- **Shoulder Elevation > 90°**: {ctx.get('shoulder_above_90_pct', 'N/A')}%
- **Severe Bending Duration**: {ctx.get('severe_bending_seconds', 'N/A')} seconds
- **Load Events**: {ctx.get('load_events', 0)}
- **Tools Detected (by object detector)**: {ctx.get('tools_detected', 'None')}
- **Rule-Based Task Classification**: {ctx.get('rule_task', 'N/A')}

## Required Output

Respond ONLY with a valid JSON object (no markdown, no code fences) using this exact schema:

{{
  "ai_task_classification": "Specific agricultural task name (e.g., 'Rice Transplanting', 'Turmeric Rhizome Harvesting', 'Maize Weeding', 'Sugarcane Cutting')",
  "ai_task_confidence": 0.85,
  "ai_crop_detected": "Crop name visible in frames or inferred (e.g., 'Rice Paddy', 'Turmeric', 'Unknown')",
  "ai_terrain_type": "Field terrain type (e.g., 'Wet Paddy Field', 'Dry Upland Plot', 'Greenhouse', 'Hillside Terrace')",
  "ai_risk_narrative": "2-3 sentence expert ergonomic risk assessment based on visual evidence AND biomechanical data. Reference specific posture concerns, joint angles, and MSD risks.",
  "ai_recommendations": [
    "Specific actionable recommendation 1",
    "Specific actionable recommendation 2",
    "Specific actionable recommendation 3"
  ],
  "ai_tools_identified": ["Tool names visible in frames (e.g., 'hand hoe', 'sickle', 'basket')"],
  "ai_severity_rating": "One of: Low, Moderate, High, Critical",
  "ai_working_conditions": "Brief observation about weather, lighting, protective equipment, field conditions visible in frames"
}}"""


# ─────────────────────────────────────────────────────
# Keyframe Selection
# ─────────────────────────────────────────────────────

def select_representative_keyframes(
    frames: List[Tuple[float, np.ndarray]],
    trunk_flexions: List[Optional[float]],
    max_frames: int = GEMINI_MAX_FRAMES_PER_WORKER,
) -> List[Tuple[float, np.ndarray]]:
    """
    Select representative keyframes that capture the full range of worker activity.

    Strategy:
        1. Always include first and last frames
        2. Include frame with maximum trunk flexion (peak bending)
        3. Include frame with minimum trunk flexion (most upright)
        4. Fill remaining slots with uniformly spaced frames

    Args:
        frames: List of (timestamp, frame_bgr) tuples.
        trunk_flexions: Corresponding trunk flexion angles (may contain None).
        max_frames: Maximum number of frames to return.

    Returns:
        Selected subset of (timestamp, frame_bgr) tuples.
    """
    if len(frames) <= max_frames:
        return frames

    selected_indices = set()

    # Always include first and last
    selected_indices.add(0)
    selected_indices.add(len(frames) - 1)

    # Find peak bending (max trunk flexion) and most upright (min trunk flexion)
    valid_flexions = [(i, f) for i, f in enumerate(trunk_flexions) if f is not None]
    if valid_flexions:
        max_idx = max(valid_flexions, key=lambda x: x[1])[0]
        min_idx = min(valid_flexions, key=lambda x: x[1])[0]
        selected_indices.add(max_idx)
        selected_indices.add(min_idx)

    # Fill remaining with uniform spacing
    remaining = max_frames - len(selected_indices)
    if remaining > 0:
        step = len(frames) / (remaining + 1)
        for i in range(1, remaining + 1):
            idx = min(int(i * step), len(frames) - 1)
            selected_indices.add(idx)

    # Sort by timestamp order and return
    sorted_indices = sorted(selected_indices)[:max_frames]
    return [frames[i] for i in sorted_indices]


def _resize_frame_for_api(frame: np.ndarray, target_size: Tuple[int, int] = GEMINI_FRAME_RESIZE) -> bytes:
    """Resize a BGR frame and encode as JPEG bytes for API transmission."""
    resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
    _, jpeg_bytes = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return jpeg_bytes.tobytes()


# ─────────────────────────────────────────────────────
# Main Analyzer Class
# ─────────────────────────────────────────────────────

class GeminiAnalyzer:
    """
    Multimodal AI analyzer using Google Gemini for agricultural ergonomic assessment.

    Usage:
        analyzer = GeminiAnalyzer(api_key="your-key")
        result = analyzer.analyze_worker(
            keyframes=[(ts, frame), ...],
            biomechanical_context={...},
        )
        print(result.ai_task_classification)
        print(result.ai_risk_narrative)
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemini analyzer.

        Args:
            api_key: Google Gemini API key. Falls back to GEMINI_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get(GEMINI_API_KEY_ENV, "")
        self.model_name = GEMINI_MODEL
        self._client = None
        self._available = False

        if self.api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
                self._available = True
                logger.info(f"Gemini AI analyzer initialized with model: {self.model_name}")
            except ImportError:
                logger.warning("google-genai package not installed. AI analysis disabled.")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini client: {e}")
        else:
            logger.info("No Gemini API key provided. AI analysis will be skipped.")

    @property
    def is_available(self) -> bool:
        """Check if Gemini AI analysis is available."""
        return self._available and self._client is not None

    def analyze_worker(
        self,
        keyframes: List[Tuple[float, np.ndarray]],
        biomechanical_context: Dict[str, Any],
    ) -> AIAnalysisResult:
        """
        Analyze a worker's keyframes with Gemini multimodal vision.

        Args:
            keyframes: List of (timestamp_seconds, frame_bgr_array) tuples.
            biomechanical_context: Dict with biomechanical measurements for prompt context.

        Returns:
            AIAnalysisResult with task classification, risk narrative, and recommendations.
        """
        if not self.is_available:
            return AIAnalysisResult(
                ai_risk_narrative="AI analysis unavailable — no API key configured.",
                ai_model_used="none",
            )

        if not keyframes:
            return AIAnalysisResult(
                ai_risk_narrative="No keyframes available for AI analysis.",
                ai_model_used=self.model_name,
            )

        try:
            return self._call_gemini(keyframes, biomechanical_context)
        except Exception as e:
            logger.error(f"Gemini AI analysis failed: {e}", exc_info=True)
            return AIAnalysisResult(
                ai_risk_narrative=f"AI analysis failed: {str(e)[:200]}",
                ai_model_used=self.model_name,
            )

    def _call_gemini(
        self,
        keyframes: List[Tuple[float, np.ndarray]],
        biomechanical_context: Dict[str, Any],
    ) -> AIAnalysisResult:
        """Execute the Gemini API call with frames and structured prompt."""
        from google import genai
        from google.genai import types

        # Build multimodal content parts
        content_parts = []

        # Add each keyframe as an image part
        for i, (timestamp, frame) in enumerate(keyframes):
            jpeg_bytes = _resize_frame_for_api(frame)
            content_parts.append(
                types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg")
            )
            content_parts.append(
                types.Part.from_text(text=f"[Frame {i+1}/{len(keyframes)} — Timestamp: {timestamp:.1f}s]")
            )

        # Add the analysis prompt
        prompt_text = _build_analysis_prompt(biomechanical_context)
        content_parts.append(types.Part.from_text(text=prompt_text))

        # Call Gemini
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=content_parts,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=GEMINI_TEMPERATURE,
                max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
            ),
        )

        # Parse structured JSON response
        return self._parse_response(response.text)

    def _parse_response(self, response_text: str) -> AIAnalysisResult:
        """Parse Gemini's JSON response into an AIAnalysisResult."""
        # Strip markdown code fences if present
        text = response_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last line (code fence markers)
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Attempt to extract JSON from mixed text
            import re
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                logger.warning(f"Could not parse Gemini response as JSON: {text[:500]}")
                return AIAnalysisResult(
                    ai_risk_narrative=text[:500],
                    ai_model_used=self.model_name,
                    ai_analysis_success=False,
                )

        return AIAnalysisResult(
            ai_task_classification=data.get("ai_task_classification", "Unknown"),
            ai_task_confidence=float(data.get("ai_task_confidence", 0.0)),
            ai_crop_detected=data.get("ai_crop_detected", "Unknown"),
            ai_terrain_type=data.get("ai_terrain_type", "Unknown"),
            ai_risk_narrative=data.get("ai_risk_narrative", ""),
            ai_recommendations=data.get("ai_recommendations", []),
            ai_tools_identified=data.get("ai_tools_identified", []),
            ai_severity_rating=data.get("ai_severity_rating", "Unknown"),
            ai_working_conditions=data.get("ai_working_conditions", ""),
            ai_model_used=self.model_name,
            ai_analysis_success=True,
        )

    def analyze_workers_batch(
        self,
        workers_data: List[Dict[str, Any]],
        progress_callback=None,
    ) -> Dict[int, AIAnalysisResult]:
        """
        Analyze multiple workers sequentially.

        Args:
            workers_data: List of dicts, each containing:
                - 'worker_id': int
                - 'keyframes': List[(timestamp, frame)]
                - 'biomechanical_context': Dict
            progress_callback: Optional callback(fraction, message)

        Returns:
            Dict mapping worker_id -> AIAnalysisResult.
        """
        results = {}
        total = len(workers_data)

        for i, wd in enumerate(workers_data):
            wid = wd["worker_id"]
            if progress_callback:
                progress_callback(
                    i / max(1, total),
                    f"🧠 AI analyzing Worker {wid} ({i+1}/{total})..."
                )

            result = self.analyze_worker(
                keyframes=wd["keyframes"],
                biomechanical_context=wd["biomechanical_context"],
            )
            results[wid] = result

        return results
