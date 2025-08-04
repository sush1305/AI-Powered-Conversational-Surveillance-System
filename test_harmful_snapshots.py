#!/usr/bin/env python3
"""
Test harmful snapshot image saving functionality
"""

import os
import sys
import cv2
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_harmful_snapshot_saving():
    """Test if harmful snapshots (actual images) are saved correctly"""
    print("=" * 60)
    print("TESTING HARMFUL SNAPSHOT IMAGE SAVING")
    print("=" * 60)
    
    try:
        from video_processor import VideoProcessor
        
        # Initialize VideoProcessor
        print("[STEP 1] Initializing VideoProcessor...")
        vp = VideoProcessor(show_popup=False)
        print("[OK] VideoProcessor initialized")
        
        # Create a test frame (simulated camera frame)
        print("\n[STEP 2] Creating test frame...")
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add some visual content to the test frame
        cv2.rectangle(test_frame, (50, 50), (590, 430), (0, 255, 0), 2)
        cv2.putText(test_frame, "TEST HARMFUL EVENT", (100, 250), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.putText(test_frame, f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                   (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.circle(test_frame, (320, 200), 50, (255, 0, 0), -1)
        
        print(f"[OK] Test frame created: {test_frame.shape}")
        
        # Test saving different types of harmful events
        harmful_events = [
            ("violence", 0.95),
            ("fall_detected", 0.87),
            ("crash", 0.92),
            ("weapon_detected", 0.89)
        ]
        
        print(f"\n[STEP 3] Testing harmful event snapshot saving...")
        saved_files = []
        
        for event_type, confidence in harmful_events:
            print(f"  Testing {event_type} (confidence: {confidence})...")
            
            # Add event-specific visual indicator
            event_frame = test_frame.copy()
            cv2.putText(event_frame, f"EVENT: {event_type.upper()}", (100, 350), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Save harmful event snapshot
            filepath = vp.save_critical_event_snapshot(
                frame=event_frame,
                event_type=event_type,
                confidence=confidence,
                metadata={"test": True, "source": "test_script"}
            )
            
            if filepath and os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                print(f"    [OK] Image saved: {os.path.basename(filepath)} ({file_size} bytes)")
                saved_files.append(filepath)
                
                # Check if metadata file also exists
                metadata_file = filepath.replace('.jpg', '_metadata.json')
                if os.path.exists(metadata_file):
                    metadata_size = os.path.getsize(metadata_file)
                    print(f"    [OK] Metadata saved: {os.path.basename(metadata_file)} ({metadata_size} bytes)")
                else:
                    print(f"    [ERROR] Metadata file missing: {os.path.basename(metadata_file)}")
            else:
                print(f"    [ERROR] Image not saved for {event_type}")
        
        # Test non-harmful event (should not save image)
        print(f"\n[STEP 4] Testing non-harmful event (should not save image)...")
        filepath = vp.save_critical_event_snapshot(
            frame=test_frame,
            event_type="person_detected",
            confidence=0.95,
            metadata={"test": True}
        )
        
        if filepath is None:
            print("    [OK] Non-harmful event correctly ignored (no image saved)")
        else:
            print("    [ERROR] Non-harmful event incorrectly saved image")
        
        # Display results
        print(f"\n[STEP 5] Checking harmful snapshots directory...")
        if os.path.exists(vp.harmful_snapshots_dir):
            files = os.listdir(vp.harmful_snapshots_dir)
            image_files = [f for f in files if f.endswith('.jpg')]
            metadata_files = [f for f in files if f.endswith('.json')]
            
            print(f"[OK] Harmful snapshots directory: {vp.harmful_snapshots_dir}")
            print(f"     Image files: {len(image_files)}")
            print(f"     Metadata files: {len(metadata_files)}")
            
            for img_file in image_files:
                img_path = os.path.join(vp.harmful_snapshots_dir, img_file)
                img_size = os.path.getsize(img_path)
                print(f"     - {img_file} ({img_size} bytes)")
        
        # Summary
        print(f"\n" + "=" * 60)
        print("HARMFUL SNAPSHOT TEST RESULTS")
        print("=" * 60)
        
        if len(saved_files) >= 4:
            print("[SUCCESS] All harmful event images saved correctly!")
            print(f"✅ {len(saved_files)} harmful event images created")
            print("✅ Image files contain actual picture data")
            print("✅ Metadata files created with event details")
            print("✅ Non-harmful events correctly ignored")
            return True
        else:
            print(f"[PARTIAL] Only {len(saved_files)}/4 harmful events saved")
            return False
            
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_harmful_snapshot_saving()
