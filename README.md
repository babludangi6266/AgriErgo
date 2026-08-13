# AgriErgo — Video-Based Farm Worker Ergonomics & Drudgery Assessment Platform

A video-based platform that automatically analyzes field-work videos and extracts objective, quantified measurements of worker posture, movement, load handling, tool usage, and work/rest patterns for ergonomic and drudgery assessment.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```

### 3. Run the FastAPI Backend
```bash
uvicorn api.main:app --reload --port 8000
```

### 4. Run Tests
```bash
python -m pytest tests/ -v
```

## Architecture

The system is organized into three processing layers:

1. **Perception Layer** — Pose estimation (YOLOv8-Pose), object detection (YOLOv8), multi-person tracking (ByteTrack)
2. **Interpretation Layer** — Posture classification, joint angles, repetition detection, activity segmentation, trip counting
3. **Analytics & Reporting Layer** — 11-parameter aggregation, REBA ergonomic scoring, report generation

## 11 Output Parameters

| # | Parameter | Description |
|---|-----------|-------------|
| 1 | Sitting | Time spent in a seated posture |
| 2 | Standing | Time spent standing upright |
| 3 | Bending | Time spent in trunk-flexed / stooped posture |
| 4 | Walking | Time spent in locomotion |
| 5 | Load carried | Instances/type of load carried |
| 6 | Repetitive movement frequency | Cycles per minute of repeated motion |
| 7 | Number of trips | Count of back-and-forth movements |
| 8 | Tools/equipment used | Identification of tools/equipment over time |
| 9 | Posture | Classified body posture and estimated joint angles |
| 10 | Continuous work duration | Length of uninterrupted work bouts |
| 11 | Rest duration | Length and frequency of rest/idle periods |

## License

MIT
