import cv2
import sys
import os

def test_camera(index=0):
    print(f"Trying to open camera index {index}...")
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # Try DirectShow first
    
    if not cap.isOpened():
        print(f"Failed with DirectShow, trying default backend...")
        cap = cv2.VideoCapture(index)  # Try default backend
    
    if not cap.isOpened():
        print(f"[X] Could not open camera at index {index}")
        return False
    
    print("[OK] Camera opened successfully!")
    print("Testing frame capture...")
    
    # Try to read a few frames
    success_count = 0
    for i in range(5):
        ret, frame = cap.read()
        if ret and frame is not None:
            success_count += 1
            print(f"  [OK] Frame {i+1}: Success ({frame.shape[1]}x{frame.shape[0]})")
            # Save test frame
            cv2.imwrite(f'test_frame_{index}.jpg', frame)
            print(f"  [OK] Saved test frame to test_frame_{index}.jpg")
        else:
            print(f"  [X] Frame {i+1}: Failed to capture")
    
    cap.release()
    print(f"Camera test completed: {success_count}/5 frames captured successfully")
    return success_count > 0

def check_camera_permissions():
    """Check Windows camera privacy settings"""
    print("\n--- Camera Permission Check ---")
    print("Please verify:")
    print("1. Windows Settings > Privacy > Camera")
    print("2. 'Allow apps to access your camera' is ON")
    print("3. 'Allow desktop apps to access your camera' is ON")
    print("4. No other apps (Zoom, Teams, etc.) are using the camera")

if __name__ == "__main__":
    print("=== Camera Diagnostic Tool ===")
    
    # Check camera permissions first
    check_camera_permissions()
    
    # Test common camera indices (0-4)
    working_cameras = []
    for i in range(5):
        print(f"\n--- Testing Camera Index {i} ---")
        if test_camera(i):
            working_cameras.append(i)
            print(f"[OK] Camera {i} is working!")
        else:
            print(f"[X] Camera {i} failed")
    
    print(f"\n=== SUMMARY ===")
    if working_cameras:
        print(f"[OK] Found {len(working_cameras)} working camera(s): {working_cameras}")
        print(f"[OK] Use camera index {working_cameras[0]} in your config")
    else:
        print("[X] No working cameras found")
        print("[X] Check camera permissions and hardware connection")
