import cv2
import os
import numpy as np

class VideoLoader:
    def __init__(self, video_path, resize_dim=(640, 480)):
        self.video_path = video_path
        self.resize_dim = resize_dim
        self.cap = None

    def __enter__(self):
        if not os.path.exists(self.video_path):
            print(f"Warning: Video file {self.video_path} not found. Using dummy blank frames.")
            self.cap = None
        else:
            self.cap = cv2.VideoCapture(self.video_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap:
            self.cap.release()

    def stream_frames(self):
        if self.cap is None:
            # Generate dummy frames for demonstration
            for i in range(100):
                # Create a moving circle to simulate an object
                frame = np.zeros((self.resize_dim[1], self.resize_dim[0], 3), dtype=np.uint8)
                cx = 50 + i * 5
                cy = 240
                gt_mask = np.zeros((self.resize_dim[1], self.resize_dim[0]), dtype=np.uint8)
                # Simulate occlusion: Object disappears between frame 40 and 60
                if not (40 < i < 60):
                    cv2.circle(frame, (cx, cy), 20, (0, 255, 0), -1)
                    cv2.circle(gt_mask, (cx, cy), 20, 255, -1)
                yield frame, i, gt_mask
        else:
            frame_idx = 0
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, self.resize_dim)
                yield frame, frame_idx, None
                frame_idx += 1
