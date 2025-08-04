#!/usr/bin/env python3
"""
Test script to verify popup display with live camera feed
"""
import cv2
import time
import threading
import numpy as np

def test_popup_with_camera():
    """Test popup window with live camera feed"""
    print("Testing popup with live camera feed...")
    
    # Open camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Could not open camera")
        return False
    
    print("[OK] Camera opened successfully")
    
    # Create window
    window_name = 'AI Surveillance - Live Feed Test'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame")
                break
            
            frame_count += 1
            
            # Add overlay text
            cv2.putText(frame, f"Live Feed - Frame {frame_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display frame
            cv2.imshow(window_name, frame)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[OK] User pressed 'q' to quit")
                break
            
            # Small delay to control frame rate
            time.sleep(0.033)  # ~30 FPS
            
    except KeyboardInterrupt:
        print("[OK] Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[OK] Cleanup completed")
    
    return True

if __name__ == "__main__":
    print("=== Popup Display Test ===")
    success = test_popup_with_camera()
    if success:
        print("[OK] Popup test completed successfully")
    else:
        print("[ERROR] Popup test failed")
