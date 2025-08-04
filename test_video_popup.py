#!/usr/bin/env python3
"""
Test script to verify video popup functionality
"""

import cv2
import numpy as np
from datetime import datetime
import time

def test_video_popup():
    """Test the video popup functionality"""
    print("Testing Video Popup Functionality...")
    print("=" * 50)
    
    # Try to open camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[X] Camera not available, using dummy video")
        use_dummy = True
    else:
        print("[OK] Camera opened successfully")
        use_dummy = False
    
    frame_counter = 0
    
    print("\n[INFO] Video popup window should open now...")
    print("[INSTRUCTIONS]:")
    print("   - Press 'q' to close the popup")
    print("   - Press 'ESC' to exit completely")
    print("=" * 50)
    
    while True:
        if use_dummy:
            # Generate dummy frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Add gradient background
            for y in range(480):
                for x in range(640):
                    frame[y, x] = [int(x * 255 / 640), int(y * 255 / 480), 128]
            
            # Add moving circle
            center_x = int(320 + 100 * np.sin(frame_counter * 0.1))
            center_y = int(240 + 50 * np.cos(frame_counter * 0.1))
            cv2.circle(frame, (center_x, center_y), 30, (0, 255, 255), -1)
            
            frame_counter += 1
        else:
            # Read from camera
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read from camera")
                break
        
        # Add overlay information
        timestamp = datetime.now()
        cv2.putText(frame, f"AI Surveillance Test - {timestamp.strftime('%H:%M:%S')}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to close popup, 'ESC' to exit", 
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display the frame
        cv2.imshow('AI Surveillance - Test Popup', frame)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[OK] Popup closed with 'q' key")
            cv2.destroyAllWindows()
            break
        elif key == 27:  # ESC key
            print("[OK] Test completed with ESC key")
            break
        
        # Small delay for dummy video
        if use_dummy:
            time.sleep(0.03)  # ~30 FPS
    
    # Cleanup
    if not use_dummy:
        cap.release()
    cv2.destroyAllWindows()
    
    print("\n[SUCCESS] Video popup test completed!")
    print("If you saw the video window, the popup functionality is working correctly.")

if __name__ == "__main__":
    test_video_popup()
