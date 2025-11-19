# SAM 2 Video Segmentation Project - Quick Start Guide

## Overview
This project implements **SAM2-Refine**, a robust video segmentation system that combines the SAM 2 model with Kalman Filter-based occlusion handling for autonomous driving scenarios.

## Project Structure
```
Megaminds IT Services/
├── src/
│   ├── video_loader.py       # Loads video frames
│   ├── sam2_model.py          # Mock SAM 2 predictor
│   ├── occlusion_handler.py   # Kalman Filter tracker
│   └── pipeline.py            # Main processing pipeline
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
```bash
cd src
python pipeline.py
```

## What Happens When You Run It?

Since no input video is provided, the system will:
1. **Generate a dummy video** with 100 frames showing a moving green circle
2. **Simulate occlusion** by removing the object between frames 40-60
3. **Process each frame** through:
   - SAM 2 Mock Predictor (detects the green circle)
   - Kalman Filter Tracker (predicts position during occlusion)
4. **Generate output video**: `output_video.avi` in the project root

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

1. ✅ Run the code to generate `output_video.avi`
2. ✅ Review all documentation files
3. 📹 Record your 15-minute video presentation using the provided script
4. 📦 Create a ZIP file containing:
   - All `src/` code files
   - `requirements.txt`
   - `README.md`
   - Documentation (Comprehensive Report, Case Study, etc.)
   - `output_video.avi`
5. 📤 Submit the ZIP file

## Contact
For questions about the implementation, refer to the inline code comments in each Python file.
