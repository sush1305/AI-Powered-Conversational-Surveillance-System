#!/usr/bin/env python3
"""
Fix dummy video creation and recording duration issues
"""
import re

def fix_dummy_video_issues():
    """Fix dummy video creation and recording duration calculation"""
    
    # Read the current file
    with open('video_processor.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    changes_made = False
    
    # Fix 1: Stop video recording when switching to dummy mode
    pattern1 = r'(\s+# If too many failures, switch to dummy mode\s+if consecutive_failures >= max_failures:\s+logger\.error\("Camera has failed consistently, switching to dummy video source"\)\s+)(# Release current camera\s+if self\.cap:\s+self\.cap\.release\(\)\s+self\.cap = None\s+)(# Switch to dummy mode\s+self\.source = "dummy"\s+logger\.info\("Starting dummy video capture loop"\)\s+self\._dummy_capture_loop\(\)\s+return  # Exit this capture loop)'
    
    replacement1 = r'''\1# Stop video recording to prevent dummy video files
                    if self.is_recording:
                        self._stop_video_recording()
                        logger.info("Video recording stopped due to camera failure")
                    
                    \2# Switch to dummy mode WITHOUT recording
                    self.source = "dummy"
                    logger.info("Starting dummy video capture loop (NO RECORDING)")
                    self._dummy_capture_loop()
                    return  # Exit this capture loop'''
    
    new_content = re.sub(pattern1, replacement1, content, flags=re.DOTALL)
    if new_content != content:
        content = new_content
        changes_made = True
        print("[OK] Fixed dummy mode video recording prevention")
    
    # Fix 2: Prevent dummy capture loop from recording
    pattern2 = r'(def _dummy_capture_loop\(self\):.*?"Generate synthetic frames when no camera is available".*?)(while self\.is_running:.*?# Create a synthetic frame.*?frame = self\._generate_dummy_frame\(frame_counter\))'
    
    replacement2 = r'''\1# IMPORTANT: Stop any video recording in dummy mode
            if self.is_recording:
                self._stop_video_recording()
                logger.info("Stopped video recording in dummy mode")
            
            \2'''
    
    new_content = re.sub(pattern2, replacement2, content, flags=re.DOTALL)
    if new_content != content:
        content = new_content
        changes_made = True
        print("[OK] Fixed dummy capture loop to prevent recording")
    
    # Fix 3: Fix recording duration calculation
    pattern3 = r'(# Update recording duration\s+current_time = datetime\.now\(\)\s+duration = \(current_time - self\.session_start\)\.total_seconds\(\)\s+session_data\["statistics"\]\["recording_duration_seconds"\] = int\(duration\))'
    
    replacement3 = r'''# Update recording duration
                current_time = datetime.now()
                if hasattr(self, 'session_start') and self.session_start:
                    duration = (current_time - self.session_start).total_seconds()
                    session_data["statistics"]["recording_duration_seconds"] = int(duration)
                else:
                    # Fallback: calculate from video start time if available
                    if hasattr(self, 'video_start_time') and self.video_start_time:
                        duration = (current_time - self.video_start_time).total_seconds()
                        session_data["statistics"]["recording_duration_seconds"] = int(duration)
                    else:
                        session_data["statistics"]["recording_duration_seconds"] = 0'''
    
    new_content = re.sub(pattern3, replacement3, content, flags=re.DOTALL)
    if new_content != content:
        content = new_content
        changes_made = True
        print("[OK] Fixed recording duration calculation")
    
    # Write the fixed content back if changes were made
    if changes_made:
        with open('video_processor.py', 'w', encoding='utf-8') as f:
            f.write(content)
        print("[OK] All fixes applied to video_processor.py")
        return True
    else:
        print("[INFO] No changes needed or patterns not found")
        return False

if __name__ == "__main__":
    print("=== Fixing Dummy Video and Recording Duration Issues ===")
    success = fix_dummy_video_issues()
    if success:
        print("[OK] Dummy video and recording duration fixes applied successfully")
    else:
        print("[INFO] No changes made")
