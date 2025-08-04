#!/usr/bin/env python3
"""
Patch script to fix popup frame update in video_processor.py
"""
import re

def fix_popup_frame_update():
    """Fix the popup frame update in video_processor.py"""
    
    # Read the current file
    with open('video_processor.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the capture loop section where frames are updated
    # Look for the pattern where current_frame is updated
    pattern = r'(\s+# Update current frame\s+with self\.frame_lock:\s+self\.current_frame = timestamped_frame\s+)'
    
    # Replacement that adds popup frame update
    replacement = r'\1\n            # Update popup frame for display\n            with self.popup_frame_lock:\n                self.popup_frame = frame.copy()\n                '
    
    # Apply the fix
    new_content = re.sub(pattern, replacement, content)
    
    # Check if the fix was applied
    if new_content != content:
        # Write the fixed content back
        with open('video_processor.py', 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("[OK] Fixed popup frame update in video_processor.py")
        return True
    else:
        print("[INFO] Popup frame update already present or pattern not found")
        return False

if __name__ == "__main__":
    print("=== Fixing Popup Frame Update ===")
    success = fix_popup_frame_update()
    if success:
        print("[OK] Popup frame update fix applied successfully")
    else:
        print("[INFO] No changes needed or pattern not found")
