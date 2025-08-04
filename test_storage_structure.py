#!/usr/bin/env python3
"""
Test script to verify the enhanced storage structure
"""

import os
import sys
import cv2
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from video_processor import VideoProcessor

def test_storage_structure():
    """Test the enhanced storage structure"""
    
    print("=" * 60)
    print("TESTING ENHANCED STORAGE STRUCTURE")
    print("=" * 60)
    
    # Initialize video processor
    print("\n[STEP 1] Initializing VideoProcessor...")
    video_processor = VideoProcessor(show_popup=False)
    
    # Check if directories were created
    print("\n[STEP 2] Checking directory creation...")
    expected_dirs = [
        video_processor.storage_dir,
        video_processor.video_dir,
        video_processor.suspicious_dir,
        video_processor.logs_dir,
        video_processor.database_dir
    ]
    
    for directory in expected_dirs:
        if os.path.exists(directory):
            print(f"[OK] {directory} - EXISTS")
        else:
            print(f"[X] {directory} - MISSING")
    
    # Test action logging
    print("\n[STEP 3] Testing action logging...")
    try:
        video_processor._log_action("test_event", 0.85, "Testing action logging functionality")
        if os.path.exists(video_processor.action_log_file):
            print(f"[OK] Action log file created: {video_processor.action_log_file}")
            
            # Read and display log content
            with open(video_processor.action_log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                print("[FILE] Log file content:")
                print("-" * 40)
                print(content)
                print("-" * 40)
        else:
            print("[X] Action log file not created")
    except Exception as e:
        print(f"[X] Action logging failed: {e}")
    
    # Test critical event detection
    print("\n[STEP 4] Testing critical event detection...")
    test_events = [
        ("motion", False),
        ("person_detected", False),
        ("fall", True),
        ("violence", True),
        ("crash", True),
        ("weapon", True),
        ("normal_activity", False)
    ]
    
    for event_type, expected_critical in test_events:
        is_critical = video_processor._is_critical_event(event_type)
        status = "[OK]" if is_critical == expected_critical else "[X]"
        print(f"{status} {event_type}: {'CRITICAL' if is_critical else 'NON-CRITICAL'}")
    
    # Test critical event snapshot saving
    print("\n[STEP 5] Testing critical event snapshot saving...")
    
    # Create test frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    test_frame[:] = (50, 100, 150)  # Fill with color
    cv2.putText(test_frame, "TEST CRITICAL EVENT", (150, 240), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Test critical event (should save snapshot)
    print("  Testing critical event (fall)...")
    snapshot_path = video_processor.save_critical_event_snapshot(
        test_frame, "fall", 0.92, {"location": "entrance", "severity": "high"}
    )
    
    if snapshot_path and os.path.exists(snapshot_path):
        print(f"[OK] Critical event snapshot saved: {snapshot_path}")
    else:
        print("[X] Critical event snapshot not saved")
    
    # Test non-critical event (should not save snapshot)
    print("  Testing non-critical event (motion)...")
    snapshot_path = video_processor.save_critical_event_snapshot(
        test_frame, "motion", 0.75, {"location": "hallway"}
    )
    
    if snapshot_path is None:
        print("[OK] Non-critical event correctly ignored (no snapshot saved)")
    else:
        print("[X] Non-critical event incorrectly saved snapshot")
    
    # Display final storage structure
    print("\n[STEP 6] Final storage structure:")
    print("=" * 60)
    
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
                    else:
                        print(f"   [DIR] {file}/")
            else:
                print("   (empty)")
        else:
            print("   (directory not found)")
    
    display_directory_contents(video_processor.video_dir, "Videos")
    display_directory_contents(video_processor.suspicious_dir, "Suspicious Events (Critical Only)")
    display_directory_contents(video_processor.logs_dir, "Action Logs")
    display_directory_contents(video_processor.database_dir, "Database")
    
    print("\n" + "=" * 60)
    print("STORAGE STRUCTURE TEST COMPLETED")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    try:
        success = test_storage_structure()
        if success:
            print("\n[SUCCESS] Enhanced storage structure is working correctly!")
        else:
            print("\n[X] Storage structure test failed!")
    except Exception as e:
        print(f"\n[ERROR] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
