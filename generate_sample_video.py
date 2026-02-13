"""
Sample Video Generator
Creates a simple test video for demonstrating the accident detection system
"""

import cv2
import numpy as np

def create_sample_video(output_path='sample_traffic.mp4', duration=10):
    """
    Create a sample video simulating traffic and an accident
    
    Args:
        output_path: Path to save the video
        duration: Duration in seconds
    """
    
    # Video properties
    width, height = 640, 480
    fps = 30
    total_frames = duration * fps
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Creating sample video: {output_path}")
    print(f"Duration: {duration}s, FPS: {fps}, Total frames: {total_frames}")
    
    # Car properties
    car1_x, car1_y = 50, height // 2
    car2_x, car2_y = width - 150, height // 2
    car_speed = 5
    
    for frame_num in range(total_frames):
        # Create black background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw road
        cv2.rectangle(frame, (0, height//2 - 50), (width, height//2 + 50), (70, 70, 70), -1)
        
        # Draw lane markings
        for i in range(0, width, 40):
            cv2.rectangle(frame, (i, height//2 - 2), (i + 20, height//2 + 2), (255, 255, 255), -1)
        
        # Simulate accident around 70% through video
        if frame_num < int(total_frames * 0.7):
            # Normal traffic - cars moving
            car1_x += car_speed
            car2_x -= car_speed
            
            # Draw cars
            cv2.rectangle(frame, (int(car1_x), car1_y - 20), 
                         (int(car1_x + 60), car1_y + 20), (0, 0, 255), -1)
            cv2.rectangle(frame, (int(car2_x), car2_y - 20), 
                         (int(car2_x + 60), car2_y + 20), (0, 255, 0), -1)
            
            # Add text
            cv2.putText(frame, "Normal Traffic", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
        else:
            # Accident scenario
            # Position cars at collision point
            collision_x = width // 2
            
            # Draw damaged cars with random positions (simulating impact)
            offset1 = np.random.randint(-5, 5)
            offset2 = np.random.randint(-5, 5)
            
            cv2.rectangle(frame, (collision_x - 30 + offset1, car1_y - 20), 
                         (collision_x + 30 + offset1, car1_y + 20), (0, 0, 255), -1)
            cv2.rectangle(frame, (collision_x + offset2, car2_y - 20), 
                         (collision_x + 60 + offset2, car2_y + 20), (0, 255, 0), -1)
            
            # Add debris (random rectangles)
            for _ in range(10):
                debris_x = collision_x + np.random.randint(-80, 80)
                debris_y = car1_y + np.random.randint(-40, 40)
                debris_size = np.random.randint(5, 15)
                color = (np.random.randint(100, 255), 
                        np.random.randint(100, 255), 
                        np.random.randint(100, 255))
                cv2.rectangle(frame, (debris_x, debris_y), 
                            (debris_x + debris_size, debris_y + debris_size), 
                            color, -1)
            
            # Add warning text
            cv2.putText(frame, "!!! ACCIDENT !!!", (width//2 - 150, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
            # Add smoke effect (gray circles)
            for i in range(3):
                smoke_x = collision_x + np.random.randint(-50, 50)
                smoke_y = car1_y + np.random.randint(-30, 30)
                smoke_radius = np.random.randint(10, 30)
                cv2.circle(frame, (smoke_x, smoke_y), smoke_radius, (100, 100, 100), -1)
        
        # Add frame number
        cv2.putText(frame, f"Frame: {frame_num}", (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Write frame
        out.write(frame)
        
        # Progress indicator
        if frame_num % 30 == 0:
            progress = (frame_num / total_frames) * 100
            print(f"Progress: {progress:.1f}%")
    
    # Release video writer
    out.release()
    print(f"\n✓ Sample video created: {output_path}")
    print(f"  - Normal traffic: 0s - {duration * 0.7:.1f}s")
    print(f"  - Accident scene: {duration * 0.7:.1f}s - {duration}s")
    print("\nUse this video to test the accident detection system!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate sample traffic video')
    parser.add_argument('--output', type=str, default='sample_traffic.mp4',
                       help='Output video path')
    parser.add_argument('--duration', type=int, default=10,
                       help='Video duration in seconds')
    
    args = parser.parse_args()
    
    create_sample_video(args.output, args.duration)
