import argparse
import os
from typing import Optional, Tuple

import cv2
import numpy as np

from video_loader import VideoLoader
from sam2_model import SAM2Predictor
from occlusion_handler import KalmanTracker
from quality_controller import QualityController
from memory_manager import MemoryManager
from metrics import MetricsLogger


def _centroid_from_mask(mask: Optional[np.ndarray]) -> Optional[Tuple[int, int]]:
    if mask is None or np.count_nonzero(mask) == 0:
        return None
    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        return None
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    return cx, cy


def _save_frame(directory: str, label: str, frame_idx: int, frame: np.ndarray) -> None:
    filename = f"{label}_frame_{frame_idx:03d}.png"
    path = os.path.join(directory, filename)
    cv2.imwrite(path, frame)


def _evaluate_quality(
    controller: Optional[QualityController], mask: np.ndarray, score: float
) -> dict:
    if controller is not None:
        return controller.evaluate(mask, score)
    mask_area = float(np.count_nonzero(mask))
    norm_area = mask_area / float(mask.size)
    quality = 0.6 * score + 0.4 * norm_area
    is_reliable = (score > 0.5) and (mask_area > 0)
    return {
        "is_reliable": is_reliable,
        "quality": quality,
        "threshold": 0.5,
        "norm_area": norm_area,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SAM2-Refine+ video segmentation pipeline"
    )
    parser.add_argument(
        "--tag", default="proposed", help="Label for artifacts/output segregation"
    )
    parser.add_argument(
        "--disable-quality",
        action="store_true",
        help="Disable adaptive quality gating (use fixed 0.5 threshold)",
    )
    parser.add_argument(
        "--disable-memory",
        action="store_true",
        help="Disable memory-based mask recovery (Kalman-only)",
    )
    parser.add_argument(
        "--video",
        default="input_video.mp4",
        help="Override input video relative to project root",
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    video_path = os.path.join(project_dir, args.video)
    output_filename = f"output_video_{args.tag}.avi"
    output_path = os.path.join(project_dir, output_filename)

    loader = VideoLoader(video_path)
    predictor = SAM2Predictor(mock=True)
    tracker = KalmanTracker()
    quality_controller = None if args.disable_quality else QualityController()
    memory_manager = None if args.disable_memory else MemoryManager(capacity=25)
    metrics_logger = MetricsLogger()

    artifacts_dir = os.path.join(project_dir, "artifacts", args.tag)
    frames_dir = os.path.join(artifacts_dir, "frames")
    metrics_dir = os.path.join(artifacts_dir, "metrics")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))

    print("Starting Video Segmentation Pipeline...")
    if args.disable_quality:
        print("Adaptive quality gating: OFF (fixed threshold)")
    else:
        print("Adaptive quality gating: ON")
    if args.disable_memory:
        print("Memory-based recovery: OFF")
    else:
        print("Memory-based recovery: ON")

    keyframes = {"tracking": None, "occlusion": None, "recovery": None}
    previous_status = ""

    with loader as video:
        for frame, idx, gt_mask in video.stream_frames():
            masks, scores = predictor.predict(frame, idx)
            primary_mask = masks[0]
            primary_score = scores[0]

            quality_info = _evaluate_quality(
                controller=quality_controller,
                mask=primary_mask,
                score=primary_score,
            )

            refined_mask, position, status = tracker.update(
                mask=primary_mask,
                score=primary_score,
                quality_info=quality_info,
                frame_idx=idx,
                frame_shape=primary_mask.shape,
                memory_manager=memory_manager,
            )

            gt_position = _centroid_from_mask(gt_mask)

            metrics_logger.log(
                frame_idx=idx,
                status=status,
                quality_info=quality_info,
                pred_mask=refined_mask,
                gt_mask=gt_mask,
                predicted_pos=position,
                gt_pos=gt_position,
            )

            colored_mask = np.zeros_like(frame)
            colored_mask[:, :, 1] = refined_mask

            alpha = 0.5
            if status.startswith("Occluded"):
                colored_mask[:, :, 1] = 0
                colored_mask[:, :, 2] = refined_mask

            output_frame = cv2.addWeighted(frame, 1, colored_mask, alpha, 0)

            cv2.putText(
                output_frame,
                f"Frame: {idx}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                output_frame,
                f"Status: {status}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                output_frame,
                f"Quality: {quality_info['quality']:.2f} / {quality_info['threshold']:.2f}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.circle(output_frame, position, 5, (0, 0, 255), -1)

            out.write(output_frame)

            if quality_info["is_reliable"] and keyframes["tracking"] is None:
                _save_frame(frames_dir, "tracking", idx, output_frame)
                keyframes["tracking"] = idx
            if status.startswith("Occluded") and keyframes["occlusion"] is None:
                _save_frame(frames_dir, "occlusion", idx, output_frame)
                keyframes["occlusion"] = idx
            if (
                status == "Tracking"
                and previous_status.startswith("Occluded")
                and keyframes["recovery"] is None
            ):
                _save_frame(frames_dir, "recovery", idx, output_frame)
                keyframes["recovery"] = idx

            previous_status = status

            if idx % 20 == 0:
                print(f"Processed Frame {idx}: {status}")

    out.release()
    metrics_path = os.path.join(metrics_dir, "run_metrics.csv")
    plot_path = os.path.join(metrics_dir, "run_metrics.png")
    metrics_logger.save_csv(metrics_path)
    metrics_logger.plot_curves(plot_path)
    print(f"Processing Complete. Output saved to {output_path}")
    print(f"Metrics saved to {metrics_path}")
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
