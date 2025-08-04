#!/usr/bin/env python3
"""
Comprehensive diagnostic for video system issues
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

def test_basic_opencv():
    """Test basic OpenCV functionality"""
    print("\n[TEST 1] Basic OpenCV Test")
    print("-" * 40)
    
    try:
        print(f"OpenCV version: {cv2.__version__}")
        
        # Test camera access
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Cannot open camera")
            return False
            
        print("[OK] Camera opened successfully")
        
        # Test frame reading
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Cannot read frame from camera")
            cap.release()
            return False
            
        print(f"[OK] Frame read successfully: {frame.shape}")
        
        # Test window creation
        cv2.namedWindow('Test Window', cv2.WINDOW_NORMAL)
        cv2.imshow('Test Window', frame)
        print("[OK] Window created and frame displayed")
        
        # Wait briefly then close
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        cap.release()
        
        print("[SUCCESS] Basic OpenCV test passed")
        return True
        
    except Exception as e:
        print(f"[ERROR] Basic OpenCV test failed: {e}")
        return False

def test_video_processor_initialization():
    """Test VideoProcessor initialization"""
    print("\n[TEST 2] VideoProcessor Initialization Test")
    print("-" * 40)
    
    try:
        from video_processor import VideoProcessor
        
        # Initialize without popup first
        print("Initializing VideoProcessor (no popup)...")
        vp = VideoProcessor(show_popup=False)
        print("[OK] VideoProcessor initialized")
        
        # Check if directories were created
        if os.path.exists(vp.storage_dir):
            print(f"[OK] Storage directory created: {vp.storage_dir}")
        else:
            print(f"[ERROR] Storage directory not created: {vp.storage_dir}")
            
        return vp
        
    except Exception as e:
        print(f"[ERROR] VideoProcessor initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_video_capture_start(vp):
    """Test video capture startup"""
    print("\n[TEST 3] Video Capture Start Test")
    print("-" * 40)
    
    try:
        print("Starting video capture...")
        success = vp.start_capture()
        
        if success:
            print("[OK] Video capture started successfully")
            
            # Wait a moment for frames to start flowing
            time.sleep(2)
            
            # Check if frames are being captured
            if hasattr(vp, 'current_frame') and vp.current_frame:
                print("[OK] Frames are being captured")
                frame_info = vp.current_frame
                print(f"     Frame shape: {frame_info['frame'].shape}")
                print(f"     Timestamp: {frame_info['timestamp']}")
                return True
            else:
                print("[ERROR] No frames being captured")
                return False
                
        else:
            print("[ERROR] Video capture failed to start")
            return False
            
    except Exception as e:
        print(f"[ERROR] Video capture test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_popup_display(vp):
    """Test popup display functionality"""
    print("\n[TEST 4] Popup Display Test")
    print("-" * 40)
    
    try:
        # Enable popup
        vp.show_popup = True
        print("Popup enabled, testing for 5 seconds...")
        
        start_time = time.time()
        popup_logged = False
        
        while time.time() - start_time < 5:
            # Check if popup logs appeared
            if hasattr(vp, '_popup_displayed'):
                popup_logged = True
                print("[OK] Popup display logged successfully")
                break
            time.sleep(0.1)
        
        if not popup_logged:
            print("[ERROR] Popup display not logged - popup may not be showing")
            
        # Disable popup
        vp.show_popup = False
        cv2.destroyAllWindows()
        
        return popup_logged
        
    except Exception as e:
        print(f"[ERROR] Popup display test failed: {e}")
        return False

def test_video_recording(vp):
    """Test video recording functionality"""
    print("\n[TEST 5] Video Recording Test")
    print("-" * 40)
    
    try:
        print("Testing video recording for 3 seconds...")
        
        # Check if recording starts automatically
        time.sleep(3)
        
        # Check if video files were created
        video_files = []
        if os.path.exists(vp.video_dir):
            video_files = [f for f in os.listdir(vp.video_dir) if f.endswith('.mp4')]
            
        if video_files:
            print(f"[OK] Video files created: {len(video_files)} files")
            for file in video_files:
                file_path = os.path.join(vp.video_dir, file)
                size = os.path.getsize(file_path)
                print(f"     {file} ({size} bytes)")
            return True
        else:
            print("[ERROR] No video files created")
            return False
            
    except Exception as e:
        print(f"[ERROR] Video recording test failed: {e}")
        return False

def test_web_frame_access(vp):
    """Test if frames are accessible for web dashboard"""
    print("\n[TEST 6] Web Frame Access Test")
    print("-" * 40)
    
    try:
        # Try to get current frame
        if hasattr(vp, 'current_frame') and vp.current_frame:
            frame_data = vp.current_frame
            frame = frame_data['frame']
            
            # Test frame encoding (for web streaming)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            
            if len(buffer) > 0:
                print(f"[OK] Frame encoded for web streaming: {len(buffer)} bytes")
                return True
            else:
                print("[ERROR] Frame encoding failed")
                return False
        else:
            print("[ERROR] No current frame available for web access")
            return False
            
    except Exception as e:
        print(f"[ERROR] Web frame access test failed: {e}")
        return False

def main():
    """Run comprehensive video system diagnostic"""
    print("=" * 60)
    print("COMPREHENSIVE VIDEO SYSTEM DIAGNOSTIC")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Basic OpenCV
    results['opencv'] = test_basic_opencv()
    
    # Test 2: VideoProcessor initialization
    vp = test_video_processor_initialization()
    results['initialization'] = vp is not None
    
    if vp:
        # Test 3: Video capture
        results['capture'] = test_video_capture_start(vp)
        
        if results['capture']:
            # Test 4: Popup display
            results['popup'] = test_popup_display(vp)
            
            # Test 5: Video recording
            results['recording'] = test_video_recording(vp)
            
            # Test 6: Web frame access
            results['web_frames'] = test_web_frame_access(vp)
        
        # Cleanup
        try:
            vp.stop_capture()
        except:
            pass
    
    # Summary
    print("\n" + "=" * 60)
    print("DIAGNOSTIC RESULTS SUMMARY")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "[OK]" if result else "[FAILED]"
        print(f"{status} {test_name.replace('_', ' ').title()}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n[SUCCESS] All video system components are working!")
    else:
        print(f"\n[ISSUES] {total_tests - passed_tests} components need fixing")
        
        # Provide specific guidance
        if not results.get('opencv', True):
            print("- Fix: OpenCV installation or camera access issues")
        if not results.get('initialization', True):
            print("- Fix: VideoProcessor initialization errors")
        if not results.get('capture', True):
            print("- Fix: Video capture startup issues")
        if not results.get('popup', True):
            print("- Fix: Popup display threading issues")
        if not results.get('recording', True):
            print("- Fix: Video recording functionality")
        if not results.get('web_frames', True):
            print("- Fix: Web dashboard frame access")

if __name__ == "__main__":
    main()
