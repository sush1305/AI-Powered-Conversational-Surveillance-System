#!/usr/bin/env python3
"""
Comprehensive test for all three critical video system fixes:
1. Popup window display
2. Web video feed streaming  
3. Video recording to files
"""

import os
import sys
import cv2
import numpy as np
import time
import threading
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_all_video_fixes():
    """Test all three critical video system fixes"""
    print("=" * 70)
    print("COMPREHENSIVE TEST - ALL THREE VIDEO SYSTEM FIXES")
    print("=" * 70)
    
    try:
        from video_processor import VideoProcessor
        
        # Initialize VideoProcessor with popup enabled
        print("[STEP 1] Initializing VideoProcessor with popup enabled...")
        vp = VideoProcessor(show_popup=True)
        print("[OK] VideoProcessor initialized")
        
        # Start video capture
        print("\n[STEP 2] Starting video capture system...")
        success = vp.start_capture()
        
        if not success:
            print("[ERROR] Video capture failed to start")
            return False
            
        print("[OK] Video capture started successfully")
        
        # Wait for system to stabilize and capture frames
        print("\n[STEP 3] Waiting for system to stabilize (10 seconds)...")
        print("         Look for popup window to appear!")
        print("         Check for video recording messages in console")
        
        for i in range(10):
            time.sleep(1)
            print(f"         Waiting... {i+1}/10 seconds")
            
            # Check if frames are being captured
            if i == 5:  # Check halfway through
                current_frame = vp.get_current_frame()
                if current_frame:
                    print("         [OK] Frames are being captured!")
                else:
                    print("         [WARNING] No frames captured yet...")
        
        # Test 1: Check if frames are being captured (fixes popup and web feed)
        print("\n[TEST 1] Frame Capture Test")
        print("-" * 40)
        
        current_frame = vp.get_current_frame()
        if current_frame:
            frame = current_frame['frame']
            timestamp = current_frame['timestamp']
            print(f"[OK] Current frame available: {frame.shape}")
            print(f"[OK] Frame timestamp: {timestamp}")
            
            # Test frame encoding for web streaming
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            if len(buffer) > 0:
                print(f"[OK] Frame can be encoded for web streaming: {len(buffer)} bytes")
            else:
                print("[ERROR] Frame encoding failed")
        else:
            print("[ERROR] No current frame available")
            
        # Test 2: Check popup display
        print("\n[TEST 2] Popup Display Test")
        print("-" * 40)
        
        if hasattr(vp, '_popup_displayed'):
            print("[OK] Popup display was attempted and logged")
        else:
            print("[WARNING] Popup display not logged - may not be visible")
            
        # Test 3: Check video recording
        print("\n[TEST 3] Video Recording Test")
        print("-" * 40)
        
        if vp.is_recording and vp.current_video_file:
            print(f"[OK] Video recording active: {vp.current_video_file}")
            
            # Check if video file exists and has content
            if os.path.exists(vp.current_video_file):
                file_size = os.path.getsize(vp.current_video_file)
                print(f"[OK] Video file exists: {file_size} bytes")
                
                if file_size > 1000:  # At least 1KB
                    print("[OK] Video file has substantial content")
                else:
                    print("[WARNING] Video file is very small")
            else:
                print("[ERROR] Video file does not exist")
        else:
            print("[ERROR] Video recording not active")
            
        # Test 4: Check recorded videos directory
        print("\n[TEST 4] Recorded Videos Directory Test")
        print("-" * 40)
        
        if os.path.exists(vp.video_dir):
            video_files = [f for f in os.listdir(vp.video_dir) if f.endswith('.mp4')]
            print(f"[OK] Video directory exists: {vp.video_dir}")
            print(f"[OK] Video files found: {len(video_files)}")
            
            for video_file in video_files:
                video_path = os.path.join(vp.video_dir, video_file)
                video_size = os.path.getsize(video_path)
                print(f"     - {video_file} ({video_size} bytes)")
        else:
            print("[ERROR] Video directory does not exist")
            
        # Test 5: Web dashboard frame access
        print("\n[TEST 5] Web Dashboard Frame Access Test")
        print("-" * 40)
        
        # Simulate web dashboard frame request
        web_frame = vp.get_current_frame()
        if web_frame:
            frame = web_frame['frame']
            
            # Test JPEG encoding (used by web dashboard)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            ret, jpeg_buffer = cv2.imencode('.jpg', frame, encode_param)
            
            if ret and len(jpeg_buffer) > 0:
                print(f"[OK] Web dashboard can access frames: {len(jpeg_buffer)} bytes JPEG")
            else:
                print("[ERROR] Web dashboard frame encoding failed")
        else:
            print("[ERROR] No frame available for web dashboard")
            
        # Stop the system
        print("\n[STEP 4] Stopping video capture system...")
        vp.stop_capture()
        cv2.destroyAllWindows()  # Close any popup windows
        print("[OK] System stopped")
        
        # Final summary
        print("\n" + "=" * 70)
        print("FINAL TEST RESULTS SUMMARY")
        print("=" * 70)
        
        # Check results
        frames_working = current_frame is not None
        popup_attempted = hasattr(vp, '_popup_displayed')
        recording_working = vp.current_video_file and os.path.exists(vp.current_video_file)
        
        print(f"[{'OK' if frames_working else 'FAILED'}] Issue 1 - Frame Capture: {'Working' if frames_working else 'Not Working'}")
        print(f"[{'OK' if popup_attempted else 'FAILED'}] Issue 2 - Popup Display: {'Attempted' if popup_attempted else 'Not Attempted'}")
        print(f"[{'OK' if recording_working else 'FAILED'}] Issue 3 - Video Recording: {'Working' if recording_working else 'Not Working'}")
        
        total_fixed = sum([frames_working, popup_attempted, recording_working])
        
        if total_fixed == 3:
            print(f"\n[SUCCESS] All 3 critical issues have been FIXED! ✅")
            print("✅ Popup window should be visible")
            print("✅ Web video feed should stream properly") 
            print("✅ Video recordings are being saved")
            return True
        else:
            print(f"\n[PARTIAL] {total_fixed}/3 issues fixed")
            if not frames_working:
                print("❌ Frame capture still not working")
            if not popup_attempted:
                print("❌ Popup display still not working")
            if not recording_working:
                print("❌ Video recording still not working")
            return False
            
    except Exception as e:
        print(f"[ERROR] Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_all_video_fixes()
