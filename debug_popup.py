#!/usr/bin/env python3
"""
Debug script to test popup functionality in the main surveillance system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from video_processor import VideoProcessor
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_main_system_popup():
    """Test popup functionality using the main VideoProcessor class"""
    
    print("=" * 60)
    print("[DEBUG] DEBUGGING VIDEO POPUP IN MAIN SURVEILLANCE SYSTEM")
    print("=" * 60)
    
    try:
        # Initialize video processor with popup enabled
        print("\n[STEP 1] Initializing VideoProcessor with popup enabled...")
        video_processor = VideoProcessor(show_popup=True)
        
        print(f"[INFO] Popup enabled: {video_processor.show_popup}")
        print(f"[INFO] Video source: {video_processor.source}")
        
        # Start video capture
        print("\n[STEP 2] Starting video capture...")
        if video_processor.start_capture():
            print("[SUCCESS] Video capture started successfully")
            
            print("\n[STEP 3] Running for 10 seconds to test popup...")
            print("[INSTRUCTIONS]:")
            print("  - Look for a video popup window")
            print("  - Press 'q' in the popup to close it")
            print("  - System will auto-stop after 10 seconds")
            print("-" * 60)
            
            # Let it run for 10 seconds
            start_time = time.time()
            while time.time() - start_time < 10:
                if not video_processor.is_running:
                    print("[WARNING] Video capture stopped unexpectedly")
                    break
                time.sleep(0.1)
            
            print("\n[STEP 4] Stopping video capture...")
            video_processor.stop_capture()
            print("[SUCCESS] Video capture stopped")
            
        else:
            print("[ERROR] Failed to start video capture")
            return False
            
    except Exception as e:
        print(f"[ERROR] Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("[RESULTS] DEBUG RESULTS:")
    print("=" * 60)
    print("[OK] If you saw a video popup window, the main system popup works!")
    print("[X] If no popup appeared, there's an issue with the main system.")
    print("[INFO] Check the console logs above for any error messages.")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    test_main_system_popup()
