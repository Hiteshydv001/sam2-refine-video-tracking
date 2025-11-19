import csv
import os
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class FrameMetrics:
    frame_idx: int
    status: str
    quality: float
    threshold: float
    norm_area: float
    iou: Optional[float]
    centroid_error: Optional[float]


class MetricsLogger:
    """Accumulates per-frame metrics and exports plots for the report."""

    def __init__(self):
        self._records: List[FrameMetrics] = []

    def log(
        self,
        frame_idx: int,
        status: str,
        quality_info: dict,
        pred_mask: np.ndarray,
        gt_mask: Optional[np.ndarray],
        predicted_pos: Tuple[int, int],
        gt_pos: Optional[Tuple[int, int]],
    ) -> None:
        iou = None
        centroid_error = None
        if gt_mask is not None:
            iou = self._compute_iou(pred_mask, gt_mask)
        if gt_pos is not None:
            centroid_error = float(
                np.linalg.norm(
                    [predicted_pos[0] - gt_pos[0], predicted_pos[1] - gt_pos[1]]
                )
            )
        record = FrameMetrics(
            frame_idx=frame_idx,
            status=status,
            quality=float(quality_info.get("quality", 0.0)),
            threshold=float(quality_info.get("threshold", 0.0)),
            norm_area=float(quality_info.get("norm_area", 0.0)),
            iou=iou,
            centroid_error=centroid_error,
        )
        self._records.append(record)

    def save_csv(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(FrameMetrics.__annotations__.keys()),
            )
            writer.writeheader()
            for record in self._records:
                writer.writerow(asdict(record))

    def plot_curves(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        frames = [r.frame_idx for r in self._records]
        ious = [r.iou for r in self._records]
        errors = [r.centroid_error for r in self._records]
        quality = [r.quality for r in self._records]
        threshold = [r.threshold for r in self._records]
        plt.figure(figsize=(10, 6))
        if any(v is not None for v in ious):
            plt.plot(frames, [v if v is not None else np.nan for v in ious], label="IoU")
        if any(v is not None for v in errors):
            plt.plot(
                frames,
                [v if v is not None else np.nan for v in errors],
                label="Centroid Error",
            )
        plt.plot(frames, quality, label="Quality", linestyle="--")
        plt.plot(frames, threshold, label="Adaptive Threshold", linestyle=":")
        plt.xlabel("Frame")
        plt.ylabel("Metric Value")
        plt.title("SAM2-Refine+ Tracking Metrics")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()

    @staticmethod
    def _compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        pred = pred_mask.astype(bool)
        gt = gt_mask.astype(bool)
        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        return float(intersection / union)
