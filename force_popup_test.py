#!/usr/bin/env python3
"""
Force popup test to diagnose why popup isn't showing in main system
"""

import cv2
import numpy as np
import time
import logging
from datetime import datetime
import threading

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def force_popup_test():
    """Force popup display to test if OpenCV works in current environment"""
    
    print("=" * 60)
    print("[FORCE POPUP TEST] Testing OpenCV popup in current environment")
    print("=" * 60)
    
    # Test 1: Basic OpenCV functionality
    print("\n[TEST 1] Basic OpenCV window test...")
    try:
        # Create a simple test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        test_image[:] = (50, 100, 150)  # Fill with color
        
        cv2.putText(test_image, "OpenCV Test Window", (200, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.namedWindow('OpenCV Test', cv2.WINDOW_NORMAL)
        cv2.imshow('OpenCV Test', test_image)
        
        print("[SUCCESS] Basic OpenCV window created")
        print("Press any key in the window to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"[ERROR] Basic OpenCV test failed: {e}")
        return False
    
    # Test 2: Camera access test
    print("\n[TEST 2] Camera access test...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[WARNING] Camera not available")
        return False
    
    print("[SUCCESS] Camera opened successfully")
    
    # Test 3: Live camera popup (similar to main system)
    print("\n[TEST 3] Live camera popup test...")
    print("This should show exactly what the main system should display")
    print("Press 'q' to close the popup")
    print("-" * 60)
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read camera frame")
                break
            
            # Add overlay (exactly like main system)
            timestamp = datetime.now()
            display_frame = frame.copy()
            
            cv2.putText(display_frame, f"AI Surveillance System - {timestamp.strftime('%H:%M:%S')}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press 'q' to close popup (system continues running)", 
                       (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add status indicator
            cv2.circle(display_frame, (display_frame.shape[1] - 30, 30), 10, (0, 255, 0), -1)
            cv2.putText(display_frame, "LIVE", (display_frame.shape[1] - 60, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Enhanced window creation (exactly like main system)
            window_name = 'AI Surveillance - Live Feed'
            
            if frame_count == 0:
                print(f"[INFO] Creating window: {window_name}")
                print(f"[INFO] Frame shape: {display_frame.shape}")
                print(f"[INFO] OpenCV version: {cv2.__version__}")
            
            # Create window with specific properties
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(window_name, 800, 600)
            cv2.moveWindow(window_name, 100, 100)
            
            # Display the frame
            cv2.imshow(window_name, display_frame)
            
            # Force window update and bring to front
            try:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
            except:
                pass  # Some OpenCV versions don't support this
            
            # Check for 'q' key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print(f"[SUCCESS] Popup closed by user at frame {frame_count}")
                break
            elif key == 27:  # ESC
                print(f"[SUCCESS] Test completed with ESC at frame {frame_count}")
                break
            
            frame_count += 1
            
            # Auto-exit after 30 seconds
            if time.time() - start_time > 30:
                print(f"[INFO] Auto-exit after 30 seconds at frame {frame_count}")
                break
                
    except Exception as e:
        print(f"[ERROR] Live camera test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    print(f"\n[RESULTS] Force popup test completed")
    print(f"[INFO] Total frames processed: {frame_count}")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = force_popup_test()
    if success:
        print("\n[CONCLUSION] OpenCV popup works fine!")
        print("The issue is in the main system's capture loop execution.")
    else:
        print("\n[CONCLUSION] OpenCV popup has issues in this environment.")
        print("This explains why the main system popup doesn't work.")
