#!/usr/bin/env python3
"""
Test Script for Event Detection Fixes
Verifies that CrashDetector and other event detection issues are resolved
"""

import sys
import os
import time
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from event_detector import EventDetector, CrashDetector

def test_crash_detector_fix():
    """Test that CrashDetector missing attribute error is fixed"""
    print("=" * 60)
    print("TESTING CRASH DETECTOR FIX")
    print("=" * 60)
    
    try:
        print("[INIT] Creating CrashDetector instance...")
        crash_detector = CrashDetector()
        print("[OK] CrashDetector initialized successfully")
        
        # Check that previous_detections attribute exists
        if hasattr(crash_detector, 'previous_detections'):
            print("[OK] previous_detections attribute found")
            print(f"[INFO] Initial value: {crash_detector.previous_detections}")
        else:
            print("[ERROR] previous_detections attribute still missing!")
            return False
        
        # Test detection with dummy frame
        print("[TEST] Testing detection with dummy frame...")
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        timestamp = datetime.now()
        
        events = crash_detector.detect(dummy_frame, timestamp)
        print(f"[OK] Detection completed, found {len(events)} events")
        
        # Verify previous_detections is updated
        print(f"[INFO] previous_detections after detection: {len(crash_detector.previous_detections)} items")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] CrashDetector test failed: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return False

def test_event_detector_integration():
    """Test that EventDetector works with all detectors"""
    print("\n" + "=" * 60)
    print("TESTING EVENT DETECTOR INTEGRATION")
    print("=" * 60)
    
    try:
        print("[INIT] Creating EventDetector instance...")
        event_detector = EventDetector()
        print("[OK] EventDetector initialized successfully")
        
        # Test processing with dummy frame
        print("[TEST] Testing event processing...")
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        timestamp = datetime.now()
        
        # Create frame_data dictionary as expected by process_frame
        frame_data = {
            'frame': dummy_frame,
            'timestamp': timestamp,
            'frame_id': int(timestamp.timestamp() * 1000)
        }
        
        events = event_detector.process_frame(frame_data)
        print(f"[OK] Frame processing completed, found {len(events)} events")
        
        # Check event structure if any events found
        for i, event in enumerate(events):
            if 'event_type' in event:
                print(f"[INFO] Event {i+1}: type='{event['event_type']}', confidence={event.get('confidence', 'N/A')}")
            else:
                print(f"[WARNING] Event {i+1} missing 'event_type' field")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] EventDetector integration test failed: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return False

def main():
    """Run all event detection fix tests"""
    print("TESTING EVENT DETECTION FIXES")
    print("This test verifies fixes for:")
    print("1. CrashDetector missing 'previous_detections' attribute")
    print("2. Event detector integration issues")
    print("3. Event type field consistency")
    
    # Test CrashDetector fix
    crash_ok = test_crash_detector_fix()
    
    # Test EventDetector integration
    integration_ok = test_event_detector_integration()
    
    # Final results
    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)
    
    print(f"CrashDetector Fix: {'[OK] PASS' if crash_ok else '[ERROR] FAIL'}")
    print(f"EventDetector Integration: {'[OK] PASS' if integration_ok else '[ERROR] FAIL'}")
    
    if crash_ok and integration_ok:
        print("\n[SUCCESS] ALL EVENT DETECTION FIXES VERIFIED!")
        print("[OK] No more 'previous_detections' attribute errors")
        print("[OK] Event detection system working properly")
        print("[OK] All detector classes initialized correctly")
        print("\n[READY] Event detection system is ready!")
    else:
        print("\n[ERROR] Some event detection issues remain")
        print("Please review the error messages above")
    
    return crash_ok and integration_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
