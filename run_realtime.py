"""
AgriErgo — Real-Time Live Body Angle & Posture Tracker CLI.

Usage:
    # Run on default webcam (Device index 0):
    python run_realtime.py

    # Run on external USB camera:
    python run_realtime.py --camera 1

    # Run real-time tracking on a video file / RTSP stream:
    python run_realtime.py --video sample.mp4

Keyboard Shortcuts:
    'q' / ESC — Exit real-time tracker
    's'       — Save snapshot of current frame
    'h'       — Toggle Biomechanical HUD overlay
    'r'       — Reset session timers and counters
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agriergo.perception.realtime_tracker import RealTimePostureTracker


def main():
    parser = argparse.ArgumentParser(
        description="AgriErgo Real-Time Live Body Angle & Posture Tracker"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0 for built-in/USB webcam)",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to video file or RTSP stream URL to process in real-time",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="Inference image resolution (default: 320 for ultra-high FPS)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Keypoint detection confidence threshold (default: 0.35)",
    )
    parser.add_argument(
        "--no-hud",
        action="store_true",
        help="Disable the on-screen glassmorphic HUD card",
    )
    args = parser.parse_args()

    # Determine video source
    if args.video is not None:
        source_desc = f"Video: {args.video}"
        cap = cv2.VideoCapture(args.video)
    else:
        source_desc = f"Camera #{args.camera}"
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        if not cap.isOpened():
            # Fallback for standard OpenCV backend
            cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print(f"❌ Error: Could not open video source ({source_desc}).")
        sys.exit(1)

    print("=" * 65)
    print("🌾 AgriErgo Real-Time Body Angle & Posture Tracking")
    print("=" * 65)
    print(f"📡 Video Source: {source_desc}")
    print(f"⚡ Model Resolution: {args.imgsz}px | Confidence: {args.conf}")
    print("⌨️  Shortcuts: [q] Quit  |  [s] Save Snapshot  |  [h] Toggle HUD  |  [r] Reset")
    print("=" * 65)

    tracker = RealTimePostureTracker(
        confidence=args.conf,
        imgsz=args.imgsz,
        enable_tracking=True,
    )

    draw_hud = not args.no_hud
    window_name = "AgriErgo — Real-Time Body Angle & Posture Tracker"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if args.video is not None:
                    # Loop video if running on file
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print("⚠️ End of stream or camera disconnected.")
                    break

            # Process frame through Real-Time Posture Tracker
            result = tracker.process_frame(
                frame,
                draw_overlay=True,
                draw_hud=draw_hud,
            )

            cv2.imshow(window_name, result.annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('h'):
                draw_hud = not draw_hud
                print(f"HUD Display: {'ON' if draw_hud else 'OFF'}")
            elif key == ord('r'):
                tracker.reset()
                print("Session reset.")
            elif key == ord('s'):
                timestamp = int(time.time())
                out_path = f"agriergo_snapshot_{timestamp}.jpg"
                cv2.imwrite(out_path, result.annotated_frame)
                print(f"📸 Snapshot saved to: {out_path}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nSession ended. Goodbye!")


if __name__ == "__main__":
    main()
