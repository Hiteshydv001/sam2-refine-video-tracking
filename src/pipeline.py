import cv2
import os
import numpy as np
from video_loader import VideoLoader
from sam2_model import SAM2Predictor
from occlusion_handler import KalmanTracker

def main():
    # 1. Setup
    video_path = "input_video.mp4" # Will use dummy if not found
    output_path = "output_video.avi"
    
    loader = VideoLoader(video_path)
    predictor = SAM2Predictor(mock=True)
    tracker = KalmanTracker()
    
    # Video Writer setup
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))
    
    print("Starting Video Segmentation Pipeline...")
    
    with loader as video:
        for frame, idx in video.stream_frames():
            # 2. SAM 2 Prediction
            masks, scores = predictor.predict(frame, idx)
            
            # For this demo, we track the primary object (index 0)
            primary_mask = masks[0]
            primary_score = scores[0]
            
            # 3. Occlusion Handling & Refinement
            refined_mask, position, status = tracker.update(primary_mask, primary_score)
            
            # 4. Visualization
            # Draw the mask overlay
            colored_mask = np.zeros_like(frame)
            colored_mask[:, :, 1] = refined_mask # Green channel
            
            # Blend
            alpha = 0.5
            if status.startswith("Occluded"):
                # Red tint for occluded/predicted
                colored_mask[:, :, 1] = 0
                colored_mask[:, :, 2] = refined_mask 
            
            output_frame = cv2.addWeighted(frame, 1, colored_mask, alpha, 0)
            
            # Draw status text
            cv2.putText(output_frame, f"Frame: {idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(output_frame, f"Status: {status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.circle(output_frame, position, 5, (0, 0, 255), -1)
            
            # Write to file
            out.write(output_frame)
            
            # Optional: Show window (commented out for headless env)
            # cv2.imshow('SAM 2 Refined', output_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            
            if idx % 20 == 0:
                print(f"Processed Frame {idx}: {status}")

    out.release()
    print(f"Processing Complete. Output saved to {output_path}")

if __name__ == "__main__":
    main()
