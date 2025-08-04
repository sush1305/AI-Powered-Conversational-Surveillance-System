#!/usr/bin/env python3
"""
Fix camera stability issues in video_processor.py
"""
import re

def fix_camera_stability():
    """Fix camera stability and frame reading issues"""
    
    # Read the current file
    with open('video_processor.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find and replace the capture loop with more robust version
    pattern = r'(def _capture_loop\(self\):.*?)(while self\.is_running:.*?)(ret, frame = self\.cap\.read\(\).*?)(if not ret:.*?consecutive_failures \+= 1.*?logger\.warning\(f"Failed to read frame \(attempt \{consecutive_failures\}/\{max_failures\}\)"\).*?)(# If too many failures, switch to dummy mode.*?if consecutive_failures >= max_failures:.*?self\._dummy_capture_loop\(\).*?return  # Exit this capture loop.*?)(# Wait a bit before trying again.*?time\.sleep\(0\.1\).*?continue)'
    
    replacement = r'''\1\2
            # Try to reinitialize camera if it fails
            ret, frame = self.cap.read()
            if not ret:
                consecutive_failures += 1
                logger.warning(f"Failed to read frame (attempt {consecutive_failures}/{max_failures})")
                
                # Try to reinitialize camera every 5 failures
                if consecutive_failures % 5 == 0 and consecutive_failures < max_failures:
                    logger.info(f"Attempting to reinitialize camera after {consecutive_failures} failures")
                    try:
                        self.cap.release()
                        time.sleep(0.5)  # Wait before reinitializing
                        
                        # Reinitialize camera
                        self.cap = cv2.VideoCapture(int(self.source), cv2.CAP_DSHOW)
                        if not self.cap.isOpened():
                            self.cap = cv2.VideoCapture(int(self.source))
                        
                        if self.cap.isOpened():
                            logger.info("Camera reinitialized successfully")
                            # Set buffer size
                            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            consecutive_failures = 0  # Reset counter
                            continue
                        else:
                            logger.warning("Camera reinitialization failed")
                    except Exception as e:
                        logger.error(f"Error reinitializing camera: {e}")
                
                \5\6'''
    
    # Apply the fix
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # Check if the fix was applied
    if new_content != content:
        # Write the fixed content back
        with open('video_processor.py', 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("[OK] Fixed camera stability in video_processor.py")
        return True
    else:
        print("[INFO] Camera stability fix not applied - pattern not found or already present")
        return False

if __name__ == "__main__":
    print("=== Fixing Camera Stability ===")
    success = fix_camera_stability()
    if success:
        print("[OK] Camera stability fix applied successfully")
    else:
        print("[INFO] No changes made")
