"""
Generates a sample synthetic 5-second video of a farm worker for testing.
"""

import cv2
import numpy as np
from pathlib import Path

output_dir = Path("data/samples")
output_dir.mkdir(parents=True, exist_ok=True)
video_path = output_dir / "sample_farm_worker.mp4"

width, height = 640, 480
fps = 30
duration_sec = 5
total_frames = fps * duration_sec

fourcc = cv2.VideoWriter_fourcc(*"avc1")
try:
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
except Exception:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

for i in range(total_frames):
    # Black background with green field line
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(frame, (0, 380), (width, height), (34, 139, 34), -1)  # Green field ground

    # Animate stick figure posture
    t = i / total_frames  # 0.0 to 1.0
    
    # Base hip position moving left to right (walking simulation)
    hip_x = int(150 + t * 300)
    hip_y = 300
    
    if t < 0.33:
        # Phase 1: Standing upright
        shoulder_x = hip_x
        shoulder_y = 200
        head_x, head_y = hip_x, 170
        knee_l_x, knee_l_y = hip_x - 15, 340
        knee_r_x, knee_r_y = hip_x + 15, 340
    elif t < 0.66:
        # Phase 2: Bending stooped forward
        shoulder_x = hip_x + 40
        shoulder_y = 240
        head_x, head_y = shoulder_x + 20, 230
        knee_l_x, knee_l_y = hip_x - 10, 340
        knee_r_x, knee_r_y = hip_x + 10, 340
    else:
        # Phase 3: Walking back
        shoulder_x = hip_x
        shoulder_y = 200
        head_x, head_y = hip_x, 170
        knee_l_x, knee_l_y = hip_x - 25, 340
        knee_r_x, knee_r_y = hip_x + 25, 340

    ankle_l_x, ankle_l_y = knee_l_x - 10, 380
    ankle_r_x, ankle_r_y = knee_r_x + 10, 380
    
    elbow_l_x, elbow_l_y = shoulder_x - 20, shoulder_y + 30
    hand_l_x, hand_l_y = elbow_l_x - 10, elbow_l_y + 30

    # Draw stick figure (head, spine, arms, legs)
    cv2.circle(frame, (head_x, head_y), 15, (255, 255, 255), -1)  # Head
    cv2.line(frame, (head_x, head_y + 15), (shoulder_x, shoulder_y), (255, 255, 255), 4)  # Neck
    cv2.line(frame, (shoulder_x, shoulder_y), (hip_x, hip_y), (255, 255, 255), 6)  # Torso
    
    # Legs
    cv2.line(frame, (hip_x, hip_y), (knee_l_x, knee_l_y), (255, 255, 255), 4)
    cv2.line(frame, (knee_l_x, knee_l_y), (ankle_l_x, ankle_l_y), (255, 255, 255), 4)
    cv2.line(frame, (hip_x, hip_y), (knee_r_x, knee_r_y), (255, 255, 255), 4)
    cv2.line(frame, (knee_r_x, knee_r_y), (ankle_r_x, ankle_r_y), (255, 255, 255), 4)
    
    # Arm
    cv2.line(frame, (shoulder_x, shoulder_y), (elbow_l_x, elbow_l_y), (255, 255, 255), 4)
    cv2.line(frame, (elbow_l_x, elbow_l_y), (hand_l_x, hand_l_y), (255, 255, 255), 4)

    # Add text label overlay
    cv2.putText(frame, f"AgriErgo Test Video - Frame {i+1}/{total_frames}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    out.write(frame)

out.release()
print(f"Sample video created at: {video_path.resolve()}")
