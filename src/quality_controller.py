import numpy as np
from collections import deque


class QualityController:
    """Tracks recent predictions to build an adaptive reliability threshold."""

    def __init__(self, window_size=15, min_samples=5):
        self.scores = deque(maxlen=window_size)
        self.areas = deque(maxlen=window_size)
        self.min_samples = min_samples

    def evaluate(self, mask, score):
        mask_area = float(np.count_nonzero(mask))
        norm_area = mask_area / float(mask.size)
        self.scores.append(score)
        self.areas.append(norm_area)
        adaptive_threshold = self._adaptive_threshold()
        quality = 0.6 * score + 0.4 * norm_area
        is_reliable = quality >= adaptive_threshold
        return {
            "is_reliable": is_reliable,
            "quality": quality,
            "threshold": adaptive_threshold,
            "norm_area": norm_area,
        }

    def _adaptive_threshold(self):
        if len(self.scores) < self.min_samples:
            return 0.35
        avg_score = float(np.mean(self.scores))
        avg_area = float(np.mean(self.areas))
        dynamic = 0.5 * avg_score + 0.4 * avg_area
        return max(0.35, min(0.85, dynamic))
