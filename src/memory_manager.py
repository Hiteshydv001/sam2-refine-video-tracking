import cv2
import numpy as np
from collections import deque


class MemoryManager:
    """Keeps high-quality masks and reinstates them during occlusions."""

    def __init__(self, capacity=20):
        self.buffer = deque(maxlen=capacity)

    def store(self, frame_idx, mask, quality):
        if quality <= 0.0 or mask is None:
            return
        if np.count_nonzero(mask) == 0:
            return
        moments = cv2.moments(mask)
        if moments["m00"] == 0:
            return
        centroid = (
            moments["m10"] / moments["m00"],
            moments["m01"] / moments["m00"],
        )
        self.buffer.append(
            {
                "frame_idx": frame_idx,
                "mask": mask.copy(),
                "quality": float(quality),
                "centroid": centroid,
            }
        )

    def retrieve(self, predicted_point, frame_shape):
        if not self.buffer or predicted_point is None:
            return None
        best_entry = None
        best_score = -np.inf
        px = float(predicted_point[0])
        py = float(predicted_point[1])
        for entry in self.buffer:
            cx, cy = entry["centroid"]
            distance = np.linalg.norm([px - cx, py - cy])
            score = entry["quality"] - 0.002 * distance
            if score > best_score:
                best_score = score
                best_entry = entry
        if best_entry is None:
            return None
        cx, cy = best_entry["centroid"]
        dx = px - cx
        dy = py - cy
        transform = np.float32([[1, 0, dx], [0, 1, dy]])
        warped = cv2.warpAffine(
            best_entry["mask"],
            transform,
            (frame_shape[1], frame_shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return warped
