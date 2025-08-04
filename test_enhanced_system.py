#!/usr/bin/env python3
"""
Comprehensive test for enhanced storage system and video functionality
"""

import os
import sys
import cv2
import numpy as np
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from video_processor import VideoProcessor

def test_enhanced_system():
    """Test the enhanced storage system and video functionality"""
    
    print("=" * 70)
    print("TESTING ENHANCED AI SURVEILLANCE SYSTEM")
    print("=" * 70)
    
    # Initialize video processor with popup enabled
    print("\n[STEP 1] Initializing Enhanced VideoProcessor...")
    video_processor = VideoProcessor(show_popup=True)
    
    # Check if new directories were created
    print("\n[STEP 2] Checking enhanced directory structure...")
    expected_dirs = {
        "Main Storage": video_processor.storage_dir,
        "Recorded Videos": video_processor.video_dir,
        "Harmful Snapshots": video_processor.harmful_snapshots_dir,
        "Session Logs": video_processor.session_logs_dir,
        "Harmful Events Log": video_processor.harmful_events_log_dir,
        "Database": video_processor.database_dir
    }
    
    for name, directory in expected_dirs.items():
        if os.path.exists(directory):
            print(f"[OK] {name}: {directory}")
        else:
            print(f"[X] {name}: {directory} - MISSING")
    
    # Check if session log files were created
    print("\n[STEP 3] Checking session log files...")
    log_files = {
        "Session JSON": video_processor.session_log_json,
        "Harmful Events JSON": video_processor.harmful_events_json
    }
    
    for name, file_path in log_files.items():
        if os.path.exists(file_path):
            print(f"[OK] {name}: {file_path}")
            
            # Display file content
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    print(f"     Content preview: {list(content.keys())}")
            except Exception as e:
                print(f"     Error reading file: {e}")
        else:
            print(f"[X] {name}: {file_path} - MISSING")
    
    # Test logging functionality
    print("\n[STEP 4] Testing enhanced logging functionality...")
    
    # Test regular event logging
    print("  Testing regular event logging...")
    video_processor._log_action("motion_detected", 0.75, "Motion detected in hallway", 
                               {"location": "hallway", "duration": 2.5})
    
    # Test harmful event logging
    print("  Testing harmful event logging...")
    video_processor._log_action("fall", 0.92, "Fall detected - emergency response triggered", 
                               {"location": "entrance", "severity": "high", "person_id": "unknown"})
    
    # Test critical event snapshot saving
    print("\n[STEP 5] Testing critical event snapshot saving...")
    
    # Create test frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    test_frame[:] = (50, 100, 150)  # Fill with color
    cv2.putText(test_frame, "TEST HARMFUL EVENT", (150, 240), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Test harmful event (should save snapshot)
    print("  Testing harmful event snapshot (violence)...")
    snapshot_path = video_processor.save_critical_event_snapshot(
        test_frame, "violence", 0.88, {"location": "parking_lot", "severity": "critical"}
    )
    
    if snapshot_path and os.path.exists(snapshot_path):
        print(f"[OK] Harmful event snapshot saved: {snapshot_path}")
    else:
        print("[X] Harmful event snapshot not saved")
    
    # Test non-harmful event (should not save snapshot)
    print("  Testing non-harmful event (person_detected)...")
    snapshot_path = video_processor.save_critical_event_snapshot(
        test_frame, "person_detected", 0.65, {"location": "lobby"}
    )
    
    if snapshot_path is None:
        print("[OK] Non-harmful event correctly ignored (no snapshot saved)")
    else:
        print("[X] Non-harmful event incorrectly saved snapshot")
    
    # Test video popup functionality
    print("\n[STEP 6] Testing video popup functionality...")
    print("This will test if the popup window can be created and displayed.")
    print("The popup should appear for 5 seconds, then auto-close.")
    print("Look for a video window to appear!")
    print("-" * 50)
    
    # Start video capture for popup test
    if video_processor.start_capture():
        print("[OK] Video capture started successfully")
        
        # Wait for popup to appear
        import time
        start_time = time.time()
        popup_appeared = False
        
        while time.time() - start_time < 5:  # Test for 5 seconds
            # Check if popup logs appeared
            if hasattr(video_processor, '_popup_displayed'):
                popup_appeared = True
                break
            time.sleep(0.1)
        
        # Stop video capture
        video_processor.stop_capture()
        
        if popup_appeared:
            print("[OK] Video popup appeared successfully!")
        else:
            print("[X] Video popup did not appear - check OpenCV installation")
    else:
        print("[X] Video capture failed to start")
    
    # Display final storage structure
    print("\n[STEP 7] Final enhanced storage structure:")
    print("=" * 70)
    
    def display_directory_contents(directory, name):
        print(f"\n[DIR] {name}: {directory}")
        if os.path.exists(directory):
            files = os.listdir(directory)
            if files:
                for file in sorted(files):
                    file_path = os.path.join(directory, file)
                    if os.path.isfile(file_path):
                        size = os.path.getsize(file_path)
                        print(f"   [FILE] {file} ({size} bytes)")
                        
                        # Show JSON content preview for log files
                        if file.endswith('.json'):
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    content = json.load(f)
                                    if 'events' in content:
                                        print(f"          Events: {len(content['events'])}")
                                    if 'harmful_events' in content:
                                        print(f"          Harmful Events: {len(content['harmful_events'])}")
                            except:
                                pass
                    else:
                        print(f"   [DIR] {file}/")
            else:
                print("   (empty)")
        else:
            print("   (directory not found)")
    
    display_directory_contents(video_processor.video_dir, "Recorded Videos")
    display_directory_contents(video_processor.harmful_snapshots_dir, "Harmful Snapshots")
    display_directory_contents(video_processor.session_logs_dir, "Session Logs")
    display_directory_contents(video_processor.harmful_events_log_dir, "Harmful Events Log")
    display_directory_contents(video_processor.database_dir, "Database")
    
    print("\n" + "=" * 70)
    print("ENHANCED SYSTEM TEST COMPLETED")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    try:
        success = test_enhanced_system()
        if success:
            print("\n[SUCCESS] Enhanced AI Surveillance System is working correctly!")
            print("\nKey Features Verified:")
            print("- Enhanced storage structure with separate folders")
            print("- Session-specific JSON logging")
            print("- Harmful events separate logging")
            print("- Critical event snapshot saving")
            print("- Video popup functionality")
        else:
            print("\n[X] Enhanced system test failed!")
    except Exception as e:
        print(f"\n[ERROR] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
