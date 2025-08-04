#!/usr/bin/env python3
"""
Simple test to verify video popup functionality
"""

import cv2
import numpy as np
from datetime import datetime
import time

def test_popup_display():
    """Test if OpenCV popup window works"""
    print("Testing Video Popup Display...")
    print("=" * 50)
    
    # Try to open camera first
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[INFO] Camera not available, using synthetic video")
        use_camera = False
    else:
        print("[INFO] Camera opened successfully")
        use_camera = True
    
    print("\n[POPUP] Opening video popup window...")
    print("[INSTRUCTIONS]:")
    print("  - You should see a video window appear")
    print("  - Press 'q' to close the window")
    print("  - Press 'ESC' to exit")
    print("=" * 50)
    
    frame_counter = 0
    
    try:
        while True:
            if use_camera:
                ret, frame = cap.read()
                if not ret:
                    print("[ERROR] Failed to read from camera")
                    break
            else:
                # Create synthetic frame
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                
                # Add colorful background
                for y in range(480):
                    for x in range(640):
                        frame[y, x] = [
                            int(128 + 127 * np.sin(frame_counter * 0.1 + x * 0.01)),
                            int(128 + 127 * np.cos(frame_counter * 0.1 + y * 0.01)),
                            128
                        ]
                
                # Add moving circle
                center_x = int(320 + 200 * np.sin(frame_counter * 0.05))
                center_y = int(240 + 100 * np.cos(frame_counter * 0.05))
                cv2.circle(frame, (center_x, center_y), 40, (0, 255, 255), -1)
                
                frame_counter += 1
            
            # Add overlay text
            timestamp = datetime.now()
            cv2.putText(frame, f"AI Surveillance Test - {timestamp.strftime('%H:%M:%S')}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to close, 'ESC' to exit", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display the frame - THIS IS THE KEY PART
            cv2.imshow('AI Surveillance - Popup Test', frame)
            
            # Wait for key press
            key = cv2.waitKey(30) & 0xFF  # 30ms delay for smooth video
            
            if key == ord('q'):
                print("\n[SUCCESS] Popup closed with 'q' key")
                break
            elif key == 27:  # ESC key
                print("\n[SUCCESS] Test completed with ESC key")
                break
            
            # Small delay for synthetic video
            if not use_camera:
                time.sleep(0.03)  # ~30 FPS
    
    except Exception as e:
        print(f"\n[ERROR] Exception occurred: {e}")
    
    finally:
        # Cleanup
        if use_camera and cap:
            cap.release()
        cv2.destroyAllWindows()
        print("\n[CLEANUP] All windows closed")
    
    print("\n[RESULT] Popup test completed!")
    print("If you saw a video window, the popup functionality works.")
    print("If no window appeared, there may be an OpenCV display issue.")

if __name__ == "__main__":
    test_popup_display()
