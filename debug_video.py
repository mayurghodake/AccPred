import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import sys
from app_prediction import AccidentPredictionEnsemble, process_video_prediction_flask

def test_video_processing():
    video_path = "accident_video.mp4"
    if not os.path.exists(video_path):
        # try sample_traffic.mp4
        video_path = "sample_traffic.mp4"
        if not os.path.exists(video_path):
            print("No test video found.")
            # Create a dummy video
            video_path = "dummy_test.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))
            for _ in range(60):
                frame = cv2.imread("non_existent") # Will be None
                # Just write black frames
                import numpy as np
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                out.write(frame)
            out.release()
            print("Created dummy video.")

    print(f"Testing with video: {video_path}")
    
    try:
        print("Initializing predictor...")
        predictor = AccidentPredictionEnsemble()
        
        print("Running process_video_prediction_flask...")
        warnings = process_video_prediction_flask(video_path, predictor, 0.5)
        
        print(f"Success! Found {len(warnings)} warnings.")
        for w in warnings:
            print(f"Frame {w['frame_number']}: {w['risk_level']} ({w['confidence']:.2f})")
            
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_video_processing()
