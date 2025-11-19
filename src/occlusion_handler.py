import numpy as np
import cv2

class KalmanTracker:
    def __init__(self):
        # Kalman Filter setup
        # State: [x, y, dx, dy] (Position and Velocity)
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        
        self.last_valid_pos = None
        self.is_occluded = False

    def update(self, mask, score, quality_info, frame_idx, frame_shape, memory_manager=None):
        """Fuse SAM2 predictions with Kalman estimates and memory-backed recovery."""
        # 1. Predict next state
        prediction = self.kf.predict()
        pred_x, pred_y = int(prediction[0]), int(prediction[1])
        
        # 2. Check measurement (SAM 2 output)
        measured_x, measured_y = None, None
        
        is_reliable = quality_info.get("is_reliable", False)

        if is_reliable and np.sum(mask) > 0:
            # Valid detection
            M = cv2.moments(mask)
            if M["m00"] != 0:
                measured_x = int(M["m10"] / M["m00"])
                measured_y = int(M["m01"] / M["m00"])
                
                # Correct the Kalman Filter
                self.kf.correct(np.array([[np.float32(measured_x)], [np.float32(measured_y)]]))
                self.last_valid_pos = (measured_x, measured_y)
                self.is_occluded = False
                if memory_manager is not None:
                    memory_manager.store(frame_idx, mask, quality_info.get("quality", score))
                return mask, (measured_x, measured_y), "Tracking"
        
        # 3. Handle Occlusion
        # If score is low, use Prediction
        self.is_occluded = True
        
        # Create a synthetic mask at the predicted location
        # In a real app, we might warp the previous mask using Optical Flow
        # Here we just draw a circle at the predicted position
        refined_mask = np.zeros_like(mask)
        if memory_manager is not None:
            memory_mask = memory_manager.retrieve((pred_x, pred_y), frame_shape)
            if memory_mask is not None and np.count_nonzero(memory_mask) > 0:
                refined_mask = memory_mask
        if np.count_nonzero(refined_mask) == 0:
            cv2.circle(refined_mask, (pred_x, pred_y), 20, 255, -1)
        
        return refined_mask, (pred_x, pred_y), "Occluded (KF Prediction)"
