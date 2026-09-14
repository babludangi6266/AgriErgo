"""
Tests for Gemini Multimodal Analyzer (agriergo.ai.gemini_analyzer).
"""

import json
import numpy as np
import pytest
from agriergo.ai.gemini_analyzer import (
    GeminiAnalyzer,
    AIAnalysisResult,
    _build_analysis_prompt,
)


def test_ai_analysis_result_defaults():
    """Test AIAnalysisResult default attributes."""
    res = AIAnalysisResult()
    assert res.ai_task_classification == "Unknown"
    assert res.ai_task_confidence == 0.0
    assert res.ai_analysis_success is False
    assert res.ai_recommendations == []
    assert res.ai_tools_identified == []


def test_analyzer_init_without_key():
    """Test initialization without API key falls back gracefully."""
    analyzer = GeminiAnalyzer(api_key="")
    assert analyzer.is_available() is False


def test_fallback_analysis():
    """Test fallback when API is not configured."""
    analyzer = GeminiAnalyzer(api_key="")
    ctx = {
        "rule_task": "Transplanting / Plantation",
        "dominant_posture": "bending",
        "avg_trunk_flexion": 55.0,
        "reba_score": 8,
        "drudgery_index": 65.0,
    }
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    keyframes = [(0.0, dummy_frame), (1.0, dummy_frame)]

    result = analyzer.analyze_worker(keyframes, ctx)
    assert result.ai_analysis_success is False
    assert result.ai_task_classification == "Transplanting / Plantation"
    assert len(result.ai_recommendations) > 0


def test_build_analysis_prompt():
    """Test prompt building with biomechanical context."""
    ctx = {
        "tracked_time_formatted": "00:15:00",
        "dominant_posture": "bending",
        "posture_distribution": {"bending": 70.0, "standing": 30.0},
        "avg_trunk_flexion": 58.2,
        "max_trunk_flexion": 75.0,
        "reba_score": 9,
        "reba_risk": "High",
        "rula_score": 6,
        "drudgery_index": 72.5,
        "rule_task": "Manual Weeding",
    }
    prompt = _build_analysis_prompt(ctx)
    assert "Manual Weeding" in prompt
    assert "58.2" in prompt
    assert "High" in prompt
    assert "ai_task_classification" in prompt


def test_parse_response_clean_json():
    """Test parsing clean JSON response."""
    analyzer = GeminiAnalyzer(api_key="")
    sample_json = json.dumps({
        "ai_task_classification": "Rice Transplanting",
        "ai_task_confidence": 0.92,
        "ai_crop_detected": "Paddy Rice",
        "ai_terrain_type": "Flooded Wet Paddy",
        "ai_risk_narrative": "Severe lumbar strain from sustained stooped posture.",
        "ai_recommendations": [
            "Use rolling field stool.",
            "Take 2-minute micro-breaks every 20 minutes."
        ],
        "ai_tools_identified": ["Seedling Tray"],
        "ai_severity_rating": "High",
        "ai_working_conditions": "Sunny, muddy wet soil"
    })

    result = analyzer._parse_response(sample_json)
    assert result.ai_analysis_success is True
    assert result.ai_task_classification == "Rice Transplanting"
    assert result.ai_task_confidence == 0.92
    assert result.ai_crop_detected == "Paddy Rice"
    assert len(result.ai_recommendations) == 2
    assert result.ai_severity_rating == "High"


def test_parse_response_markdown_fenced_json():
    """Test parsing JSON wrapped in markdown code fences."""
    analyzer = GeminiAnalyzer(api_key="")
    fenced_json = """```json
{
  "ai_task_classification": "Turmeric Harvesting",
  "ai_task_confidence": 0.88,
  "ai_crop_detected": "Turmeric",
  "ai_terrain_type": "Furrow Bed",
  "ai_risk_narrative": "Repeated digging shocks lower back.",
  "ai_recommendations": ["Use long-handle hoe."],
  "ai_tools_identified": ["Digging Hoe"],
  "ai_severity_rating": "Moderate",
  "ai_working_conditions": "Dry soil"
}
```"""
    result = analyzer._parse_response(fenced_json)
    assert result.ai_analysis_success is True
    assert result.ai_task_classification == "Turmeric Harvesting"
    assert result.ai_crop_detected == "Turmeric"


def test_keyframe_selection():
    """Test keyframe sampling selection."""
    analyzer = GeminiAnalyzer(api_key="")
    frames = [(float(i), np.zeros((50, 50, 3), dtype=np.uint8)) for i in range(30)]
    selected = analyzer._select_representative_keyframes(frames, max_frames=8)
    assert len(selected) <= 8
    assert len(selected) > 0
