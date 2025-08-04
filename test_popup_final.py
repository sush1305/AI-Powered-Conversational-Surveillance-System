#!/usr/bin/env python3
"""
Final popup test with enhanced debugging
"""

import cv2
import numpy as np
from datetime import datetime
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_popup_with_debugging():
    """Test popup with detailed debugging information"""
    
    print("=" * 60)
    print("[FINAL TEST] Video Popup Functionality")
    print("=" * 60)
    
    # Test OpenCV installation
    print(f"[INFO] OpenCV version: {cv2.__version__}")
    
    # Test camera access
    print("\n[STEP 1] Testing camera access...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[WARNING] Camera not available, using synthetic video")
        use_camera = False
    else:
        print("[SUCCESS] Camera opened successfully")
        use_camera = True
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"[INFO] Camera resolution: {width}x{height}")
        print(f"[INFO] Camera FPS: {fps}")
    
    print("\n[STEP 2] Testing popup window display...")
    print("[INSTRUCTIONS] Look for a video window to appear!")
    print("Press 'q' to close the window when you see it.")
    print("-" * 60)
    
    frame_count = 0
    start_time = time.time()
    popup_created = False
    
    try:
        while time.time() - start_time < 15:  # Run for 15 seconds
            if use_camera:
                ret, frame = cap.read()
                if not ret:
                    print("[ERROR] Failed to read camera frame")
                    break
            else:
                # Create synthetic frame
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                
                # Add animated background
                t = time.time() - start_time
                for y in range(480):
                    for x in range(640):
                        frame[y, x] = [
                            int(128 + 100 * np.sin(t + x * 0.01)),
                            int(128 + 100 * np.cos(t + y * 0.01)),
                            128
                        ]
                
                # Add moving elements
                center_x = int(320 + 200 * np.sin(t * 2))
                center_y = int(240 + 100 * np.cos(t * 2))
                cv2.circle(frame, (center_x, center_y), 30, (0, 255, 255), -1)
            
            # Add overlay text
            timestamp = datetime.now()
            cv2.putText(frame, f"AI Surveillance Test - {timestamp.strftime('%H:%M:%S')}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Frame: {frame_count}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to close", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add live indicator
            cv2.circle(frame, (frame.shape[1] - 30, 30), 10, (0, 255, 0), -1)
            cv2.putText(frame, "LIVE", (frame.shape[1] - 70, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Create and show window
            if not popup_created:
                cv2.namedWindow('AI Surveillance - Final Test', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('AI Surveillance - Final Test', 800, 600)
                popup_created = True
                print(f"[SUCCESS] Popup window created at frame {frame_count}")
            
            cv2.imshow('AI Surveillance - Final Test', frame)
            
            # Check for key press
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                print(f"[SUCCESS] Popup closed by user at frame {frame_count}")
                break
            elif key == 27:  # ESC
                print(f"[SUCCESS] Test completed with ESC at frame {frame_count}")
                break
            
            frame_count += 1
            
            # Progress indicator
            if frame_count % 30 == 0:  # Every second at 30 FPS
                elapsed = time.time() - start_time
                print(f"[PROGRESS] Frame {frame_count}, Time: {elapsed:.1f}s")
    
    except Exception as e:
        print(f"[ERROR] Exception during popup test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if use_camera and cap:
            cap.release()
        cv2.destroyAllWindows()
        print(f"\n[CLEANUP] Test completed after {frame_count} frames")
    
    print("\n" + "=" * 60)
    print("[RESULTS] Final Test Results:")
    print("=" * 60)
    
    if popup_created:
        print("[SUCCESS] Popup window was created successfully!")
        print("[INFO] If you saw the video window, popup functionality works.")
    else:
        print("[FAILED] Popup window was not created.")
        print("[INFO] There may be an issue with OpenCV display.")
    
    print(f"[INFO] Total frames processed: {frame_count}")
    print(f"[INFO] OpenCV version: {cv2.__version__}")
    print("=" * 60)

if __name__ == "__main__":
    test_popup_with_debugging()
