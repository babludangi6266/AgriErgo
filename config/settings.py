"""
AgriErgo — Central Configuration

All tunable parameters, model paths, thresholds, and REBA lookup tables
are centralized here for easy adjustment during prototype validation.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
RESULTS_DIR = DATA_DIR / "results"
MODELS_DIR = DATA_DIR / "models"

# Ensure runtime directories exist
for _dir in [UPLOADS_DIR, RESULTS_DIR, MODELS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Model Configuration
# ──────────────────────────────────────────────
POSE_MODEL = "yolov8n-pose.pt"        # Nano variant — CPU-friendly
DETECTION_MODEL = "yolov8n.pt"         # COCO-pretrained object detector
TRACKER_CONFIG = "bytetrack.yaml"      # ByteTrack for persistent IDs

FRAME_SAMPLE_FPS = 5                   # Sample 5 frames per second default
MAX_OCCLUSION_INTERPOLATION_FRAMES = 15 # Up to 15 sampled frames (6 seconds) interpolation
SUPPORTED_FORMATS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

# High-Speed Optimization Settings
FAST_YOLO_IMGSZ = 320                   # 320px downscaling for 4x faster CPU neural network execution
OBJECT_DETECTION_STRIDE = 5             # Run object detector every 5th frame to cut 40% CPU inference time

# ISO 11226 Cumulative Ergonomic Shift Exposure Limits
ISO11226_SEVERE_STOOPING_ANGLE = 60.0    # Bending angle >60 deg is severe
ISO11226_MAX_CUMULATIVE_STOOPING_MINS = 4.0 # Sustained >4 mins per half hour triggers hazard warning

def get_adaptive_fps(duration_seconds: float, speed_mode: str = "Balanced Fast") -> float:
    """
    Auto-tune sample FPS for instant execution (<3-8s) on short, 10-minute, and 30-minute videos.
    """
    if "Lightning" in speed_mode:
        if duration_seconds < 180:        # < 3 mins
            return 0.5
        elif duration_seconds < 900:       # 3 - 15 mins
            return 0.25
        else:                              # > 15 mins (25-30 min shift)
            return 0.15                   # ~270 frames total across 30 mins!
    elif "Precision" in speed_mode:
        if duration_seconds < 180:
            return 3.0
        elif duration_seconds < 900:
            return 1.5
        else:
            return 1.0
    else:  # Balanced Fast default
        if duration_seconds < 180:        # < 3 mins
            return 1.0
        elif duration_seconds < 900:       # 3 - 15 mins
            return 0.5
        else:                              # > 15 mins (25-30 min shift)
            return 0.25                   # ~450 frames total across 30 mins!

# ──────────────────────────────────────────────
# Confidence Thresholds
# ──────────────────────────────────────────────
POSE_CONFIDENCE = 0.3                  # Minimum keypoint confidence
DETECTION_CONFIDENCE = 0.35            # Minimum object detection confidence
TRACKING_CONFIDENCE = 0.25             # Minimum tracking confidence

# ──────────────────────────────────────────────
# COCO Keypoint Indices
# ──────────────────────────────────────────────
KP_NOSE = 0
KP_LEFT_EYE = 1
KP_RIGHT_EYE = 2
KP_LEFT_EAR = 3
KP_RIGHT_EAR = 4
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW = 7
KP_RIGHT_ELBOW = 8
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
KP_LEFT_KNEE = 13
KP_RIGHT_KNEE = 14
KP_LEFT_ANKLE = 15
KP_RIGHT_ANKLE = 16

# ──────────────────────────────────────────────
# Posture Classification Thresholds (degrees)
# ──────────────────────────────────────────────
TRUNK_FLEXION_MILD_BENDING = 20.0      # Trunk flexion 20°-60° → mild bending (ISO 11226)
TRUNK_FLEXION_BENDING = 30.0           # General bending threshold
TRUNK_FLEXION_SEVERE_BENDING = 60.0    # Trunk flexion > 60° → severe stooping
TRUNK_FLEXION_UPRIGHT = 15.0           # Trunk flexion < 15° → upright
HIP_ANGLE_SITTING = 120.0              # Hip angle < 120° → sitting/squatting candidate
HIP_ANGLE_STANDING = 160.0             # Hip angle > 160° → standing candidate
KNEE_ANGLE_SQUATTING = 100.0           # Knee angle < 100° with low hip → squatting

# Walking detection: ankle displacement between frames (pixels)
WALKING_DISPLACEMENT_THRESHOLD = 15.0  # Pixels moved between sampled frames
WALKING_MIN_CONSECUTIVE = 3            # Minimum consecutive frames with movement

# ──────────────────────────────────────────────
# Temporal Smoothing & Deadzone Filtering
# ──────────────────────────────────────────────
POSTURE_SMOOTHING_WINDOW = 5           # Confidence-weighted window over N frames
MIN_BOUT_DURATION = 1.0                # Minimum activity bout duration (seconds)
REST_STILLNESS_THRESHOLD = 5.0         # Keypoint movement below this → still (pixels)
REST_MIN_DURATION = 5.0                # Minimum rest duration (seconds)
POSITION_DEADZONE_RADIUS = 25.0        # Ignore spatial jitter within 25px radius

# ──────────────────────────────────────────────
# Repetitive Motion Detection
# ──────────────────────────────────────────────
REPETITION_FREQ_MIN = 0.2             # Min frequency (Hz) — ~12 cycles/min
REPETITION_FREQ_MAX = 3.0             # Max frequency (Hz) — ~180 cycles/min
REPETITION_PEAK_PROMINENCE = 5.0       # Minimum peak prominence (degrees)

# ──────────────────────────────────────────────
# Trip Detection
# ──────────────────────────────────────────────
TRIP_DIRECTION_CHANGE_ANGLE = 90.0     # Direction change > 90° → potential trip reversal
TRIP_MIN_DISPLACEMENT = 50.0           # Minimum displacement before reversal counts (pixels)
TRIP_SUSTAIN_FRAMES = 5                # Direction must be sustained for N frames

# ──────────────────────────────────────────────
# COCO Classes Relevant to Farm Work
# ──────────────────────────────────────────────
# Mapping of COCO class IDs to human-readable farm-relevant labels
FARM_RELEVANT_COCO_CLASSES = {
    0: "person",
    24: "backpack",
    26: "handbag",
    28: "suitcase",       # Proxy for bags/containers
    43: "knife",          # Proxy for sickle/cutting tools
    76: "scissors",       # Proxy for pruning tools
    58: "potted plant",   # Vegetation context
    1: "bicycle",         # Transport
    2: "car",             # Vehicles
    3: "motorcycle",      # Vehicles
    5: "bus",             # Vehicles
    7: "truck",           # Vehicles/tractors
    56: "chair",          # Seating
    57: "couch",          # Seating
    73: "book",           # Clipboard/records
    39: "bottle",         # Water/supplies
}

# ──────────────────────────────────────────────
# REBA Scoring Tables
# ──────────────────────────────────────────────

# Trunk posture score (based on flexion angle)
REBA_TRUNK_SCORES = [
    # (min_angle, max_angle, score)
    (0, 5, 1),       # Upright
    (5, 20, 2),      # 0-20° flexion
    (20, 60, 3),     # 20-60° flexion
    (60, 180, 4),    # >60° flexion
]

# Neck posture score
REBA_NECK_SCORES = [
    (0, 20, 1),      # 0-20° flexion
    (20, 180, 2),    # >20° flexion
]

# Legs score (based on knee flexion)
REBA_LEGS_SCORES = [
    (0, 30, 1),      # Bilateral weight bearing / walking
    (30, 60, 2),     # 30-60° flexion
    (60, 180, 3),    # >60° flexion (deep squat)
]

# Upper arm score
REBA_UPPER_ARM_SCORES = [
    (0, 20, 1),      # 0-20° flexion/extension
    (20, 45, 2),     # 20-45°
    (45, 90, 3),     # 45-90°
    (90, 180, 4),    # >90°
]

# Lower arm score
REBA_LOWER_ARM_SCORES = [
    (60, 100, 1),    # 60-100° flexion
    (0, 60, 2),      # <60° or >100°
    (100, 180, 2),
]

# Wrist score
REBA_WRIST_SCORES = [
    (0, 15, 1),      # 0-15° flexion/extension
    (15, 180, 2),    # >15°
]

# REBA Table A: Trunk × Neck × Legs → Score A
# [trunk_score - 1][neck_score - 1][legs_score - 1]
REBA_TABLE_A = [
    # Trunk = 1
    [[1, 2, 3, 4], [2, 3, 4, 5]],      # Neck=1, Neck=2 × Legs=1-4
    # Trunk = 2
    [[2, 3, 4, 5], [3, 4, 5, 6]],
    # Trunk = 3
    [[2, 4, 5, 6], [4, 5, 6, 7]],
    # Trunk = 4
    [[3, 5, 6, 7], [5, 6, 7, 8]],
    # Trunk = 5
    [[4, 6, 7, 8], [6, 7, 8, 9]],
]

# REBA Table B: Upper Arm × Lower Arm × Wrist → Score B
# [upper_arm_score - 1][lower_arm_score - 1][wrist_score - 1]
REBA_TABLE_B = [
    # Upper arm = 1
    [[1, 2], [2, 3]],
    # Upper arm = 2
    [[1, 2], [3, 4]],
    # Upper arm = 3
    [[3, 4], [5, 5]],
    # Upper arm = 4
    [[4, 5], [5, 6]],
    # Upper arm = 5
    [[6, 7], [7, 8]],
    # Upper arm = 6
    [[7, 8], [8, 9]],
]

# REBA Table C: Score A × Score B → Final REBA Score
REBA_TABLE_C = [
    #  B=1  2  3  4  5  6  7  8  9  10 11 12
    [1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7],    # A=1
    [1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8],    # A=2
    [2, 3, 3, 3, 4, 5, 6, 7, 7, 8, 8, 8],    # A=3
    [3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9],    # A=4
    [4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9, 9],    # A=5
    [6, 6, 6, 7, 8, 8, 9, 9, 10, 10, 10, 10], # A=6
    [7, 7, 7, 8, 9, 9, 9, 10, 10, 11, 11, 11], # A=7
    [8, 8, 8, 9, 10, 10, 10, 10, 10, 11, 11, 11], # A=8
    [9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12], # A=9
    [10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12], # A=10
    [11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12], # A=11
    [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12], # A=12
]

# REBA Risk Levels
REBA_RISK_LEVELS = [
    (1, 1, "Negligible", "None necessary"),
    (2, 3, "Low", "May be necessary"),
    (4, 7, "Medium", "Necessary"),
    (8, 10, "High", "Necessary soon"),
    (11, 15, "Very High", "Necessary NOW"),
]

# ──────────────────────────────────────────────
# API Configuration
# ──────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
MAX_UPLOAD_SIZE_MB = 2000              # 2 GB max upload

# ──────────────────────────────────────────────
# Streamlit Configuration
# ──────────────────────────────────────────────
STREAMLIT_PAGE_TITLE = "AgriErgo — Farm Worker Ergonomics Assessment"
STREAMLIT_PAGE_ICON = "🌾"
