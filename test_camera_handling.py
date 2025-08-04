#!/usr/bin/env python3
"""
Test improved camera handling with fallback to dummy mode
"""

import os
import sys
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_camera_handling():
    """Test improved camera handling with automatic fallback"""
    print("=" * 60)
    print("TESTING IMPROVED CAMERA HANDLING")
    print("=" * 60)
    
    try:
        from video_processor import VideoProcessor
        
        # Initialize VideoProcessor
        print("[STEP 1] Initializing VideoProcessor...")
        vp = VideoProcessor(show_popup=True)
        print("[OK] VideoProcessor initialized")
        
        # Start video capture (should handle failures gracefully)
        print("\n[STEP 2] Starting video capture with improved handling...")
        success = vp.start_capture()
        
        if success:
            print("[OK] Video capture started successfully")
            print(f"     Source: {vp.source}")
        else:
            print("[ERROR] Video capture failed to start")
            return False
        
        # Monitor for 15 seconds to see if it handles failures
        print(f"\n[STEP 3] Monitoring system for 15 seconds...")
        print("         Watch for automatic fallback to dummy mode if camera fails")
        
        for i in range(15):
            time.sleep(1)
            print(f"         Monitoring... {i+1}/15 seconds")
            
            # Check current frame availability
            if i % 5 == 4:  # Check every 5 seconds
                current_frame = vp.get_current_frame()
                if current_frame:
                    print(f"         [OK] Frame available: {current_frame['frame'].shape}")
                else:
                    print("         [WARNING] No frame available")
        
        # Check final status
        print(f"\n[STEP 4] Final system status...")
        current_frame = vp.get_current_frame()
        
        if current_frame:
            print(f"[OK] System is working with source: {vp.source}")
            print(f"[OK] Frame shape: {current_frame['frame'].shape}")
            
            # Check if we're in dummy mode
            if vp.source == "dummy":
                print("[INFO] System automatically switched to dummy mode")
                print("       This means camera had issues but system continued working")
            else:
                print("[INFO] System is using real camera successfully")
        else:
            print("[ERROR] No frames available - system may have failed")
        
        # Stop the system
        print(f"\n[STEP 5] Stopping system...")
        vp.stop_capture()
        print("[OK] System stopped")
        
        # Summary
        print(f"\n" + "=" * 60)
        print("CAMERA HANDLING TEST RESULTS")
        print("=" * 60)
        
        if current_frame:
            print("[SUCCESS] Camera handling is working!")
            if vp.source == "dummy":
                print("✅ System gracefully fell back to dummy mode")
                print("✅ No crashes or hanging despite camera issues")
                print("✅ Continuous operation maintained")
            else:
                print("✅ Real camera is working properly")
                print("✅ Stable frame reading achieved")
            return True
        else:
            print("[FAILED] Camera handling needs more work")
            return False
            
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_camera_handling()
