"""
AgriErgo — Real-Time Live Body Angle & Posture Tracker CLI.

Usage:
    # Run on default webcam (Device index 0):
    python run_realtime.py

    # Run with audible hazard alerts (880Hz tone on severe stoop / overhead reach):
    python run_realtime.py --sound

    # Run with live video recording to MP4:
    python run_realtime.py --record

    # Run with live recording, sound, and auto-export to CSV and PDF:
    python run_realtime.py --sound --record --export-csv --export-pdf

    # Run on external USB camera:
    python run_realtime.py --camera 1

    # Run real-time tracking on a video file / RTSP stream:
    python run_realtime.py --video sample.mp4 --sound --export-pdf

Keyboard Shortcuts:
    'q' / ESC — Exit real-time tracker (triggers CSV/PDF auto-export if enabled)
    's'       — Save snapshot of current frame
    'h'       — Toggle Biomechanical HUD overlay
    'r'       — Reset session timers and counters
    'm'       — Mute / Unmute audible alerts
"""

import argparse
import sys
import threading
import time
from pathlib import Path

import cv2

# Platform-specific sound support
try:
    import winsound
except ImportError:
    winsound = None

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agriergo.perception.realtime_tracker import RealTimePostureTracker


def play_hazard_tone(frequency: int = 880, duration_ms: int = 250):
    """Play audible hazard beep asynchronously on a background daemon thread."""
    if winsound is not None:
        try:
            winsound.Beep(frequency, duration_ms)
        except Exception:
            pass


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
    parser.add_argument(
        "--sound",
        action="store_true",
        help="Enable audible warning alerts for severe stooping (>60°) and overhead arms (>90°)",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record the live annotated stream to an MP4 video file in data/results/",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Export all frame telemetry records to CSV upon session exit",
    )
    parser.add_argument(
        "--export-pdf",
        action="store_true",
        help="Generate and save an ergonomic assessment PDF report upon session exit",
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

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps_prop = cap.get(cv2.CAP_PROP_FPS) or 20.0
    if fps_prop <= 0 or fps_prop > 60:
        fps_prop = 20.0

    print("=" * 68)
    print("🌾 AgriErgo Real-Time Body Angle & Posture Tracking")
    print("=" * 68)
    print(f"📡 Video Source   : {source_desc} ({width}x{height} @ {fps_prop:.1f} FPS)")
    print(f"⚡ Model Config   : {args.imgsz}px inference | Confidence threshold: {args.conf}")
    print(f"🔊 Audio Alerts   : {'ENABLED (880Hz Hazard Tone)' if args.sound else 'Disabled (pass --sound to enable)'}")
    print(f"🔴 MP4 Recording  : {'ACTIVE' if args.record else 'Disabled (pass --record to enable)'}")
    print(f"📊 CSV Export     : {'ENABLED on exit' if args.export_csv else 'Disabled'}")
    print(f"📄 PDF Report     : {'ENABLED on exit' if args.export_pdf else 'Disabled'}")
    print("-" * 68)
    print("⌨️  Shortcuts: [q] Quit | [s] Snapshot | [h] Toggle HUD | [m] Toggle Mute | [r] Reset")
    print("=" * 68)

    tracker = RealTimePostureTracker(
        confidence=args.conf,
        imgsz=args.imgsz,
        enable_tracking=True,
    )

    # Initialize MP4 recording if requested
    if args.record:
        rec_path = tracker.start_video_recording(fps=fps_prop, frame_size=(width, height))
        print(f"🔴 Live session recording initialized: {rec_path}")

    draw_hud = not args.no_hud
    audio_enabled = args.sound
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

            # Trigger audible hazard alarm if enabled
            if audio_enabled and result.audio_alert_triggered:
                threading.Thread(target=play_hazard_tone, daemon=True).start()
                if result.audio_alert_message:
                    print(f"🚨 [AUDIO ALERT] {result.audio_alert_message}")

            cv2.imshow(window_name, result.annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('h'):
                draw_hud = not draw_hud
                print(f"HUD Display: {'ON' if draw_hud else 'OFF'}")
            elif key == ord('m'):
                audio_enabled = not audio_enabled
                print(f"Audio Hazard Tone: {'UNMUTED' if audio_enabled else 'MUTED'}")
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

        # Stop active MP4 recording
        if args.record:
            rec_file = tracker.stop_video_recording()
            if rec_file:
                print(f"\n🎬 Live video recording finalized: {rec_file}")

        # Auto-export CSV telemetry
        if args.export_csv:
            csv_file = tracker.export_session_csv()
            print(f"📊 Session telemetry CSV saved: {csv_file}")

        # Auto-export PDF report
        if args.export_pdf:
            pdf_bytes = tracker.export_session_pdf(source_desc=source_desc)
            print(f"📄 Ergonomic PDF report generated successfully ({len(pdf_bytes):,} bytes).")

        print("\nSession ended. Goodbye!")


if __name__ == "__main__":
    main()

