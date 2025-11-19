# 15-Minute Presentation Script – SAM2-Refine+

## Slide 1 – Title & Context (1 min)
- Introduce yourself and the internship assignment requirements.
- State project title: “SAM2-Refine+: Adaptive Memory-Aware Video Segmentation.”
- Outline agenda: motivation, literature review, method, experiments, comparative analysis, documentation assets.

## Slide 2 – Problem Motivation (1 min)
- Highlight occlusion challenges in autonomous perception.
- Mention Yin et al. (2025) baseline: SAM2 + Kalman but lacking adaptive gating/memory pruning.
- Present research questions (RQ1–RQ3).

## Slide 3 – Literature Review Highlights (2 min)
- Bullet key themes: SAM/SAM2 promptability, motion-driven tracking (SORT/Deep SORT), memory networks (STM, XMem).
- Reference 3–4 key citations with DOIs (Kirillov et al. 2023; Oh et al. 2022; Cheng & Schwing 2022; Videnovic et al. 2025).
- Emphasize identified gap: need for analytics-friendly, adaptive SAM2 extension.

## Slide 4 – Proposed Architecture (2 min)
- Display architecture diagram from report (loader → SAM2 → Quality Controller ↔ Kalman ↔ Memory → visualization).
- Explain adaptive confidence gating (rolling window) and memory manager (quality-weighted buffer + affine warp).
- Mention CLI flags for experimentation (`--tag`, `--disable-quality`, `--disable-memory`).

## Slide 5 – Algorithm Walkthrough (2 min)
- Step through one frame lifecycle: prediction, quality evaluation, Kalman correction, memory fallback, logging.
- Call out where code lives (`quality_controller.py`, `memory_manager.py`, `occlusion_handler.py`).

## Slide 6 – Implementation Evidence (2 min)
- Show screenshot of terminal commands:
  ```powershell
  & ".venv\Scripts\python.exe" src\pipeline.py --tag proposed
  & ".venv\Scripts\python.exe" src\pipeline.py --tag baseline_kf --disable-quality --disable-memory
  & ".venv\Scripts\python.exe" scripts\summarize_metrics.py artifacts\baseline_kf artifacts\proposed
  ```
- Highlight generated artifacts folders, videos (`output_video_<tag>.avi`), and metrics CSV/PNG.

## Slide 7 – Visual Results (2 min)
- Display three key frames (`tracking_frame_000`, `occlusion_frame_041`, `recovery_frame_060`).
- Discuss color overlays (green vs. red) and status annotations.
- Mention adaptive threshold text overlay.

## Slide 8 – Quantitative Comparison (1 min)
- Present metric table comparing baseline vs. proposed.
- Emphasize reduced threshold (0.4339 vs. 0.5) without IoU regression.
- Explain expectation of larger gains on noisy real-world datasets.

## Slide 9 – Journal & Documentation Plan (1 min)
- Summarize five target journals (Sensors, IEEE Access, Journal of Real-Time Image Processing, Machine Vision and Applications, Multimedia Tools and Applications).
- Note that the research report contains 26 DOI-backed references with several from the target venues.
- Mention packaged deliverables: research report, case study, scripts, metrics, output video.

## Slide 10 – Future Work & Call to Action (1 min)
- Next steps: evaluate on KITTI/Bibox sequences, integrate uncertainty quantification, optimize for embedded hardware.
- Invite questions, mention that the ZIP will include code, docs, metrics, screenshots, and the recorded presentation.
