# AgriErgo — Video-Based Agricultural Ergonomics & Drudgery Assessment Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io)
[![YOLOv8-Pose](https://img.shields.io/badge/Ultralytics-YOLOv8--Pose-00599C.svg?logo=ultralytics&logoColor=white)](https://docs.ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/Tests-29%2F29%20Passed-brightgreen.svg)]()

**AgriErgo** is an AI-powered computer vision and biomechanical analysis platform designed specifically for agricultural ergonomics. It automates the assessment of field worker postures, repetitive motions, physical fatigue, and musculoskeletal disorder (MSD) risks directly from monocular video footage without requiring wearable motion capture sensors.

---

## 🛠️ Technology Stack

AgriErgo combines modern deep learning computer vision with validated occupational biomechanics and high-performance asynchronous web backends:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             AGRIERGO TECH STACK                             │
├───────────────────────┬─────────────────────────────────────────────────────┤
│ Computer Vision & AI  │ • Ultralytics YOLOv8-Pose (COCO 17-Keypoint Body)   │
│                       │ • Ultralytics YOLOv8 (Tool & Load Object Detection) │
│                       │ • ByteTrack / BoT-SORT (Multi-Worker ID Tracking)   │
│                       │ • Spatial-Temporal Tracklet Stitching (Hungarian)   │
├───────────────────────┼─────────────────────────────────────────────────────┤
│ Video Processing      │ • OpenCV 4.x (Fault-tolerant Sequential Stream Grab)│
│                       │ • HEVC / H.265 / MPEG-TS / 4K Decoder Pipeline      │
├───────────────────────┼─────────────────────────────────────────────────────┤
│ Biomechanics & Math   │ • Vector Geometry & Trigonometric Angle Solver      │
│                       │ • SciPy (FFT Repetition & Frequency Peak Analysis)  │
│                       │ • NumPy (Joint Kinematics & Coordinate Transforms)  │
├───────────────────────┼─────────────────────────────────────────────────────┤
│ Ergonomic Standards   │ • REBA (Rapid Entire Body Assessment)               │
│                       │ • RULA (Rapid Upper Limb Assessment)                │
│                       │ • Revised NIOSH Lifting Equation (RWL, LI, L5/S1)   │
│                       │ • ISO 11226 (Static Posture Ergonomic Limits)       │
│                       │ • Agricultural Drudgery Index (ADI 0–100 Scale)     │
├───────────────────────┼─────────────────────────────────────────────────────┤
│ Web & Visualization   │ • Streamlit (Interactive Multi-Worker Dashboard)    │
│                       │ • FastAPI & Uvicorn (Asynchronous REST API)         │
│                       │ • Plotly Express & Graph Objects (Radar / Gauges)   │
│                       │ • ReportLab (Publication-Ready PDF Generator)       │
├───────────────────────┼─────────────────────────────────────────────────────┤
│ Testing & DevOps      │ • pytest & pytest-cov (29/29 Unit & Integration)    │
│                       │ • Git & GitHub Actions CI/CD                        │
└───────────────────────┴─────────────────────────────────────────────────────┘
```

---

## 🌟 Key Features

### 1. ⏱️ 1-Hour Activity Standardisation Engine
- Mathematically normalizes tracked durations and counts to a **standardized 1-hour (3600s) activity window**:
  $$\text{Scaling Factor } S = \frac{3600}{\text{Tracked Seconds}}$$
- Provides standardized hourly metrics:
  - Standardized Posture Durations (e.g. `00:24:24` squatting / hr)
  - Hourly Repetitive Cycles (e.g. `1,728 cycles/hr`)
  - Hourly Load Events & Field Trips
  - 1-Hour Standardized Work-to-Rest Ratio

### 2. 📐 Geometric Posture & Action Classification
Accurately categorizes human biomechanical states into **7 primary agricultural actions** using calibrated joint angle rules:
- **Standing**: Hip, knee, and ankle align near vertical ($\theta_{\text{knee}} \ge 160^\circ$, $\theta_{\text{hip}} \ge 150^\circ$).
- **Sitting**: Bent hips & knees with low ankle elevation ($\theta_{\text{hip}} < 110^\circ$, $\theta_{\text{knee}} < 120^\circ$, pelvis lowered).
- **Bending / Stooping**: Severe forward trunk flexion ($\theta_{\text{trunk}} \ge 45^\circ$, upright or semi-extended knees $\theta_{\text{knee}} \ge 130^\circ$).
- **Squatting / Kneeling**: Deep knee and hip flexion ($\theta_{\text{knee}} < 120^\circ$, $\theta_{\text{hip}} < 120^\circ$).
- **Walking / Locomotion**: Ankle vector spatial displacement ($> 2.5\times$ bounding box width over rolling window).
- **Load Carrying**: Spatio-temporal IoU intersection between worker bounding box and carried objects (crates, bags, backpacks).
- **Work Bouts & Rest Periods**: Automatic segmentation of continuous active bouts vs. idle micro-recovery periods.

### 3. 💪 Arm Postural Study (Upper Limb Hazard Analysis)
- **Shoulder Elevation Angle**: Computes upper arm abduction/flexion angle relative to trunk midline.
- **Elbow Flexion Angle**: Computes interior angle between upper arm and forearm.
- **Hazard Exposure Metrics**:
  - `% Time Shoulder > 45°` (Moderate upper limb strain)
  - `% Time Shoulder > 90°` (Severe overhead fatigue & rotator cuff risk)
  - Classified Arm Postural Risk (`Low`, `Moderate`, `High`, `Very High`)
- **HUD Skeleton Overlays**: Real-time color-coded angle overlays directly rendered onto processed video.

### 4. 📊 Continuous Agricultural Drudgery Index (ADI)
Evaluates composite farm labor strain across **5 scientific pillars** (0–100 score & percentage):
1. **Postural Strain ($W_1 = 30\%$)**: Weighted composite of severe trunk bending, squatting, and static standing.
2. **Repetitive Movement Strain ($W_2 = 25\%$)**: Scaled from movement frequency ($0\text{--}90\text{ CPM}$).
3. **Work Duration & Rest Deficit ($W_3 = 20\%$)**: Penalizes unbroken continuous work bouts ($> 30\text{ mins}$) without recovery.
4. **Load & Tool Burden ($W_4 = 15\%$)**: Load carrying frequency and equipment handling weight.
5. **Arm & Shoulder Strain ($W_5 = 10\%$)**: Shoulder elevation duration above $45^\circ$ and $90^\circ$.

### 5. 📑 Validated Ergonomic & Biomechanical Standards
- **REBA (Rapid Entire Body Assessment)**: 1–15 body postural score.
- **RULA (Rapid Upper Limb Assessment)**: 1–7 upper limb posture score.
- **Revised NIOSH Lifting Equation**: Calculates Recommended Weight Limit (RWL), Lifting Index (LI), and **L5/S1 Lumbar Disc Compression Force ($N$)**.
- **ISO 11226 Standard**: Evaluates static posture shift exposure limits against ergonomic threshold violations.

### 6. 👥 Robust Multi-Worker Tracking & Tracklet Stitching
- Employs **ByteTrack** and **BoT-SORT** for spatial-temporal multi-person tracking.
- **Tracklet Stitcher**: Automatically stitches fragmented IDs caused by occlusions, field crops, or workers exiting and re-entering the camera frame using Hungarian bipartite graph matching.

### 7. 🎬 Fault-Tolerant High-Resolution Video Decoder
- Robust sequential stream grab sampling (`cap.grab()`) prevents HEVC (H.265), MPEG-TS, and 4K bitstream seek corruption (`PPS/POC errors`).
- Adaptive frame subsampling maintains real-time throughput on long agricultural field recordings.

---

## 📋 11 Standardized Output Parameters

| # | Parameter | Description |
|:---:|:---|:---|
| **1** | **Sitting** | Time and percentage spent in seated posture ($< 110^\circ$ hip flexion). |
| **2** | **Standing** | Time spent standing upright ($\theta_{\text{knee}} \ge 160^\circ$). |
| **3** | **Bending / Stooping** | Time spent in stooped/bent posture ($\theta_{\text{trunk}} \ge 45^\circ$) with severe bending tracking. |
| **4** | **Walking / Locomotion** | Distance traveled, locomotion time, and pixel trajectory. |
| **5** | **Load Carried** | Frequency and duration of load carrying events based on object intersection. |
| **6** | **Repetitive Movement Frequency** | Motion cycles per minute (CPM), dominant frequency (Hz), and primary joint. |
| **7** | **Trips & Distance** | Count of back-and-forth transit bouts and cumulative pixel distance. |
| **8** | **Tools / Equipment Used** | Object detection log of tools (hoe, sickle, backpack, crate, shears). |
| **9** | **Posture & Angle Study** | Dominant posture, joint angles time series, and arm postural study metrics. |
| **10** | **Continuous Work Duration** | Longest continuous work bout duration and work segment count. |
| **11** | **Rest Duration & Recovery** | Total rest time, micro-break count, and average rest bout duration. |

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10 or higher
- NVIDIA GPU with CUDA support (recommended for real-time inference, CPU fully supported)

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/babludangi6266/AgriErgo.git
cd AgriErgo

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch Interactive Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser:
1. Upload a field video (`.mp4`, `.mov`, `.avi`, `.mkv`).
2. Select processing mode (**Lightning Fast**, **Balanced Fast**, or **High Precision Research**).
3. View real-time KPIs, 1-hour standardized matrices, arm postural risk gauges, 5-pillar ADI radar charts, and download publication-ready PDF reports.

### 4. Launch FastAPI REST Backend
```bash
uvicorn api.main:app --reload --port 8000
```
Interactive Swagger API documentation available at **[http://localhost:8000/docs](http://localhost:8000/docs)**.

### 5. Run Unit & Integration Test Suite
```bash
python -m pytest tests/ -v
```

---

## 🏗️ System Architecture

```
                                  Input Video File
                                         │
                                         ▼
   ┌───────────────────────────────────────────────────────────────────────────┐
   │ 1. PERCEPTION LAYER                                                       │
   │    ├── VideoProcessor (Sequential Stream Sampling, HEVC/MPEG-TS Handler)  │
   │    ├── PoseEstimator (YOLOv8-Pose: 17 Keypoints @ Confidence > 0.3)       │
   │    ├── ObjectDetector (YOLOv8: Tools, Loads, Crates, Equipment)           │
   │    ├── ByteTrack / BoT-SORT (Persistent Multi-Person ID Association)      │
   │    └── TrackletStitcher (Spatial-Temporal Hungarian Identity Re-linking)   │
   └─────────────────────────────────────┬─────────────────────────────────────┘
                                         │
                                         ▼
   ┌───────────────────────────────────────────────────────────────────────────┐
   │ 2. INTERPRETATION LAYER                                                   │
   │    ├── JointAngles (Trunk Flexion, Knee, Hip, Shoulder & Elbow Vectors)   │
   │    ├── PostureClassifier (7 Geometric Postures + Temporal Smoothing)      │
   │    ├── RepetitionDetector (SciPy FFT Peak Detection, CPM & Frequency)     │
   │    ├── ActivitySegmenter (Work Bouts vs. Rest & Recovery Segmentation)    │
   │    ├── TripCounter (Field Transit Bouts & Cumulative Pixel Distance)      │
   │    └── TaskClassifier (Universal Agricultural Task Auto-Inference)        │
   └─────────────────────────────────────┬─────────────────────────────────────┘
                                         │
                                         ▼
   ┌───────────────────────────────────────────────────────────────────────────┐
   │ 3. ANALYTICS & REPORTING LAYER                                            │
   │    ├── HourlyStandardiser (Mathematical Normalization to 1-Hour Baseline) │
   │    ├── DrudgeryCalculator (5-Pillar ADI 0–100 Continuous Score & Fatigue) │
   │    ├── ErgonomicScorer (REBA Score + ISO 11226 Shift Exposure Check)      │
   │    ├── RULAScorer (Rapid Upper Limb Assessment Score & Action Levels)     │
   │    ├── NIOSHCalculator (RWL, Lifting Index, L5/S1 Spinal Compression)     │
   │    ├── ParameterAggregator (11-Parameter Unified Worker Data Model)       │
   │    └── PDFGenerator & ReportGenerator (JSON, CSV, & PDF Report Export)    │
   └───────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
AgriErgo/
├── agriergo/
│   ├── analytics/
│   │   ├── drudgery_index.py        # 5-Pillar Agricultural Drudgery Index
│   │   ├── niosh_calculator.py      # NIOSH RWL & L5/S1 Compression Force
│   │   ├── parameter_aggregator.py  # 11-Parameter Aggregation Engine
│   │   ├── pdf_generator.py         # Publication-Quality PDF Exporter
│   │   ├── report_generator.py      # Structured JSON & Summary Serialization
│   │   ├── rula_scorer.py           # Rapid Upper Limb Assessment Scorer
│   │   ├── standardiser.py          # 1-Hour Activity Standardization Engine
│   │   └── time_series.py           # Posture & Angle Temporal Aggregators
│   ├── interpretation/
│   │   ├── activity_segmenter.py    # Work Bout & Rest Segmentation
│   │   ├── joint_angles.py          # 2D Biomechanical Vector Angle Solver
│   │   ├── posture_classifier.py    # 7 Geometric Posture Classification
│   │   ├── repetition_detector.py   # FFT Joint Cycle Frequency Analysis
│   │   ├── task_classifier.py       # Universal Agricultural Task Classifier
│   │   └── trip_counter.py          # Field Transit & Locomotion Counter
│   ├── perception/
│   │   ├── annotator.py             # Skeleton & Joint HUD Video Overlays
│   │   ├── object_detector.py       # Tool & Load Object Recognition
│   │   ├── pose_estimator.py        # YOLOv8-Pose Multi-Person Estimator
│   │   ├── tracklet_stitcher.py     # Spatial-Temporal ID Stitching
│   │   └── video_processor.py       # Stream Grabber & Subsampling Engine
│   ├── pipeline.py                  # End-to-End AgriErgo Orchestration Engine
│   └── models/                      # Deep Learning Weight Checkpoints
├── api/
│   └── main.py                      # FastAPI REST API Backend
├── config/
│   └── settings.py                  # Ergonomic Thresholds & System Constants
├── dashboard/
│   └── app.py                       # Streamlit Interactive Web Application
├── tests/                           # Comprehensive Pytest Suite (29 Tests)
├── requirements.txt                 # Python Dependencies
├── implementation_plan.md           # Engineering & Architecture Blueprint
└── README.md                        # Documentation & User Guide
```

---

## 📜 Publication & Citation

If you use AgriErgo in your research or agricultural ergonomic assessments, please cite:

```bibtex
@software{agriergo2026,
  author = {AgriErgo Engineering Team},
  title = {AgriErgo: Video-Based Farm Worker Ergonomics & Drudgery Assessment Platform},
  year = {2026},
  url = {https://github.com/babludangi6266/AgriErgo}
}
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
