"""
Improved Sample Video Generator
Creates a video with VERY CLEAR accident that's easy to detect
"""

import cv2
import numpy as np

def create_improved_video(output_path='accident_video.mp4', duration=15):
    """
    Create a sample video with clear accident simulation
    
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
    
    print(f"Creating accident detection video: {output_path}")
    print(f"Duration: {duration}s, FPS: {fps}, Total frames: {total_frames}")
    
    # Define phases
    normal_phase_end = int(total_frames * 0.6)  # 60% normal traffic
    collision_phase = int(total_frames * 0.65)  # Collision moment
    aftermath_start = collision_phase
    
    print(f"\nVideo Structure:")
    print(f"  Normal Traffic: Frame 0 - {normal_phase_end}")
    print(f"  Collision: Frame {collision_phase}")
    print(f"  Aftermath: Frame {aftermath_start} - {total_frames}")
    
    # Car initial positions
    car1_start_x = 50
    car2_start_x = width - 100
    car_y = height // 2
    car_speed = 3
    
    for frame_num in range(total_frames):
        # Create background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (40, 40, 40)  # Dark gray background
        
        # Draw road
        road_color = (60, 60, 60)
        cv2.rectangle(frame, (0, car_y - 60), (width, car_y + 60), road_color, -1)
        
        # Draw lane markings
        for i in range(0, width, 50):
            cv2.rectangle(frame, (i, car_y - 3), (i + 25, car_y + 3), (200, 200, 200), -1)
        
        # PHASE 1: NORMAL TRAFFIC
        if frame_num < normal_phase_end:
            # Cars moving normally
            car1_x = car1_start_x + (frame_num * car_speed)
            car2_x = car2_start_x - (frame_num * car_speed)
            
            # Draw car 1 (blue, moving right)
            cv2.rectangle(frame, (int(car1_x), car_y - 25), 
                         (int(car1_x + 70), car_y + 25), (255, 100, 0), -1)
            cv2.rectangle(frame, (int(car1_x), car_y - 25), 
                         (int(car1_x + 70), car_y + 25), (255, 255, 255), 2)
            
            # Draw car 2 (green, moving left)
            cv2.rectangle(frame, (int(car2_x), car_y - 25), 
                         (int(car2_x + 70), car_y + 25), (0, 255, 100), -1)
            cv2.rectangle(frame, (int(car2_x), car_y - 25), 
                         (int(car2_x + 70), car_y + 25), (255, 255, 255), 2)
            
            # Status text
            cv2.putText(frame, "NORMAL TRAFFIC", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # PHASE 2: COLLISION MOMENT
        elif frame_num < aftermath_start + 15:
            collision_x = width // 2
            
            # Flash effect at collision moment
            if frame_num == collision_phase:
                frame[:, :] = (255, 255, 255)  # White flash
            
            # Calculate collision animation
            frames_since_collision = frame_num - collision_phase
            shake = np.random.randint(-8, 8) if frames_since_collision < 10 else 0
            
            # Draw damaged/collided cars
            # Car 1 (now red - damaged)
            car1_final_x = collision_x - 40 + shake
            cv2.rectangle(frame, (car1_final_x, car_y - 25), 
                         (car1_final_x + 70, car_y + 25), (0, 0, 255), -1)
            
            # Car 2 (also red - damaged)
            car2_final_x = collision_x + 10 + shake
            cv2.rectangle(frame, (car2_final_x, car_y - 25), 
                         (car2_final_x + 70, car_y + 25), (0, 0, 200), -1)
            
            # Add debris (LOTS of it)
            np.random.seed(frame_num)  # Consistent debris per frame
            for i in range(30):
                debris_x = collision_x + np.random.randint(-100, 100)
                debris_y = car_y + np.random.randint(-50, 50)
                debris_size = np.random.randint(3, 12)
                debris_color = (
                    np.random.randint(150, 255),
                    np.random.randint(0, 100),
                    np.random.randint(0, 50)
                )
                cv2.circle(frame, (debris_x, debris_y), debris_size, debris_color, -1)
            
            # Smoke effect (gray particles)
            for i in range(15):
                smoke_x = collision_x + np.random.randint(-60, 60)
                smoke_y = car_y + np.random.randint(-40, 40)
                smoke_size = np.random.randint(15, 35)
                smoke_alpha = np.random.randint(80, 150)
                cv2.circle(frame, (smoke_x, smoke_y), smoke_size, 
                          (smoke_alpha, smoke_alpha, smoke_alpha), -1)
            
            # Draw impact lines
            for i in range(8):
                angle = i * 45
                length = 80 if frames_since_collision < 5 else 40
                end_x = int(collision_x + length * np.cos(np.radians(angle)))
                end_y = int(car_y + length * np.sin(np.radians(angle)))
                cv2.line(frame, (collision_x, car_y), (end_x, end_y), 
                        (0, 255, 255), 3 if frames_since_collision < 5 else 1)
            
            # BIG RED WARNING TEXT
            cv2.putText(frame, "!!! ACCIDENT !!!", (width//2 - 180, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            cv2.putText(frame, "!!! COLLISION !!!", (width//2 - 180, height - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            
            # Add exclamation marks around
            for i in range(4):
                x_pos = 50 + i * 150
                cv2.putText(frame, "!", (x_pos, height//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        
        # PHASE 3: AFTERMATH
        else:
            collision_x = width // 2
            
            # Stationary damaged cars
            cv2.rectangle(frame, (collision_x - 40, car_y - 25), 
                         (collision_x + 30, car_y + 25), (0, 0, 180), -1)
            cv2.rectangle(frame, (collision_x + 10, car_y - 25), 
                         (collision_x + 80, car_y + 25), (0, 0, 180), -1)
            
            # Persistent debris
            np.random.seed(42)  # Fixed seed for consistent debris
            for i in range(25):
                debris_x = collision_x + np.random.randint(-90, 90)
                debris_y = car_y + np.random.randint(-45, 45)
                debris_size = np.random.randint(3, 10)
                cv2.circle(frame, (debris_x, debris_y), debris_size, (100, 100, 100), -1)
            
            # Warning text (blinking)
            if frame_num % 20 < 10:
                cv2.putText(frame, "ACCIDENT SCENE", (width//2 - 150, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        # Frame info
        cv2.putText(frame, f"Frame: {frame_num}", (10, height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Write frame
        out.write(frame)
        
        # Progress
        if frame_num % 30 == 0:
            progress = (frame_num / total_frames) * 100
            print(f"Progress: {progress:.0f}%", end='\r')
    
    out.release()
    print(f"\n\n✓ Video created successfully: {output_path}")
    print(f"\nVideo Timeline:")
    print(f"  00:00 - {normal_phase_end/fps:05.1f}s : Normal traffic")
    print(f"  {collision_phase/fps:05.1f}s          : COLLISION!")
    print(f"  {aftermath_start/fps:05.1f}s - {duration:05.1f}s : Accident aftermath")
    print(f"\n⚠️  Accident clearly visible from frame {collision_phase} onwards")
    print(f"\nUse this video with threshold 0.3-0.5 for best results!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate accident detection test video')
    parser.add_argument('--output', type=str, default='accident_video.mp4',
                       help='Output video path')
    parser.add_argument('--duration', type=int, default=15,
                       help='Video duration in seconds (default: 15)')
    
    args = parser.parse_args()
    
    create_improved_video(args.output, args.duration)
