# SAM 2 Video Segmentation Project - Quick Start Guide

## Overview
This project implements **SAM2-Refine+**, a robust video segmentation system that augments SAM 2 with adaptive quality gating, memory-backed occlusion recovery, and automated evaluation utilities for autonomous driving scenarios.

<div align="center">
    <a href="https://youtu.be/IPCUVS6_EUg">
        <img src="https://img.youtube.com/vi/IPCUVS6_EUg/0.jpg" alt="SAM2 Demo Video" width="800"/>
    </a>
    <br>
    <b>▶ Click to Watch the SAM2-Refine+ Demo</b>
</div>


## Project Structure
```
Megaminds IT Services/
├── src/
│   ├── video_loader.py         # Frame streamer + synthetic ground truth
│   ├── sam2_model.py           # Mock SAM 2 predictor
│   ├── occlusion_handler.py    # Kalman tracker with memory hooks
│   ├── quality_controller.py   # Adaptive confidence gating
│   ├── memory_manager.py       # Quality-aware mask buffer
│   ├── metrics.py              # Per-frame metric logger
│   └── pipeline.py             # CLI-enabled orchestration
├── scripts/
│   └── summarize_metrics.py    # Aggregates experiment metrics
├── docs/
│   ├── research_report.md      # Literature review + algorithm write-up
│   ├── case_study.md           # Business-focused case analysis
│   └── presentation_script.md  # 15-minute narration outline
├── artifacts/                  # Auto-generated videos, frames, plots (per run)
├── requirements.txt
└── README.md
```

## Installation & Setup

### Step 1: Install Dependencies
Open a terminal in the project directory and run:
```bash
pip install -r requirements.txt
```

This will install:
- `opencv-python` (for video processing)
- `numpy` (for numerical operations)

### Step 2: Run the Pipeline
Use the virtual environment created above or your preferred interpreter:
```powershell
# Proposed SAM2-Refine+ configuration
& ".venv\Scripts\python.exe" src\pipeline.py --tag proposed

# Baseline Kalman-only ablation
& ".venv\Scripts\python.exe" src\pipeline.py --tag baseline_kf --disable-quality --disable-memory

# Summarize metrics for report tables
& ".venv\Scripts\python.exe" scripts\summarize_metrics.py artifacts\baseline_kf artifacts\proposed
```

## What Happens When You Run It?

By default the system will:
1. **Generate a dummy video** (100 frames) with a moving green circle.
2. **Simulate occlusion** by removing the object between frames 41–60.
3. **Process each frame** through:
   - SAM2 mock predictor (HSV-based segmentation proxy).
   - Quality controller (adaptive threshold unless disabled).
   - Kalman tracker with optional memory recovery.
4. **Generate run-specific videos** such as `output_video_proposed.avi` in the project root and store diagnostics under `artifacts/<tag>`.

## Understanding the Output

The output video will show:
- **Green overlay**: Object is visible and tracked by SAM 2
- **Red overlay**: Object is occluded, position predicted by Kalman Filter
- **Status text**: Shows current frame number and tracking status
- **Red dot**: Shows the tracked center position

### Key Frames to Watch:
- **Frames 0-40**: Normal tracking (Green, "Tracking")
- **Frames 41-60**: Occlusion handling (Red, "Occluded (KF Prediction)")
- **Frames 61-100**: Re-acquisition (Green, "Tracking")

## Using Your Own Video

To test with your own video:
1. Place your video file in the project root
2. Rename it to `input_video.mp4` (or edit line 10 in `pipeline.py`)
3. Run the pipeline again

## Documentation

All research documentation is available in the artifacts:
- **Comprehensive Report**: Full research paper with literature review
- **Case Study**: Autonomous driving application
- **Video Script**: Presentation guide for your 15-minute video
- **Journal Suggestions**: Target journals and 25+ references

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'cv2'`
**Solution**: Run `pip install opencv-python`

**Issue**: Video player can't open `.avi` file
**Solution**: Use VLC Media Player or convert using:
```bash
ffmpeg -i output_video.avi output_video.mp4
```

## Next Steps for Assignment Submission

1. ✅ Run baseline and proposed pipelines to produce artifacts.
2. ✅ Export metric summaries via `scripts/summarize_metrics.py`.
3. ✅ Review documentation in `docs/` (research report, case study, presentation script).
4. 📹 Record a ~15-minute presentation using `docs/presentation_script.md`.
5. 📦 Zip the repository including `src/`, `scripts/`, `docs/`, `artifacts/`, `requirements.txt`, `README.md`, and generated videos.
6. 📤 Upload the ZIP together with the presentation recording (e.g., Google Drive link).

## Contact
For questions about the implementation, refer to the inline code comments in each Python file.
