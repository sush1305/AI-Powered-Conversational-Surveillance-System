#!/usr/bin/env python3
"""
Fix array ambiguity errors and recording duration issues
"""
import re

def fix_array_ambiguity_errors():
    """Fix array ambiguity errors in event_detector.py"""
    
    # Read the current file
    with open('event_detector.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    changes_made = False
    
    # Common array comparison patterns that cause ambiguity
    patterns_to_fix = [
        # Fix numpy array comparisons
        (r'if\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*==\s*None:', r'if \1 is None:'),
        (r'if\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*!=\s*None:', r'if \1 is not None:'),
        (r'if\s+([a-zA-Z_][a-zA-Z0-9_]*\.shape):', r'if len(\1) > 0:'),
        (r'if\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*and\s+([a-zA-Z_][a-zA-Z0-9_]*\.shape):', r'if \1 is not None and len(\2) > 0:'),
        (r'if\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*and\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*!=\s*None:', r'if \1 is not None and \2 is not None:'),
        # Fix frame comparisons
        (r'if\s+(frame):', r'if \1 is not None:'),
        (r'if\s+(current_frame):', r'if \1 is not None:'),
        (r'if\s+(previous_frame):', r'if \1 is not None:'),
        # Fix detection result comparisons
        (r'if\s+(results):', r'if results is not None and len(results) > 0:'),
        (r'if\s+(detections):', r'if detections is not None and len(detections) > 0:'),
    ]
    
    for pattern, replacement in patterns_to_fix:
        new_content = re.sub(pattern, replacement, content)
        if new_content != content:
            content = new_content
            changes_made = True
            print(f"[OK] Fixed pattern: {pattern}")
    
    # Write the fixed content back if changes were made
    if changes_made:
        with open('event_detector.py', 'w', encoding='utf-8') as f:
            f.write(content)
        print("[OK] Array ambiguity fixes applied to event_detector.py")
        return True
    else:
        print("[INFO] No array ambiguity patterns found to fix")
        return False

def fix_recording_duration():
    """Fix recording duration calculation in video_processor.py"""
    
    # Read the current file
    with open('video_processor.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    changes_made = False
    
    # Fix session_start initialization
    pattern1 = r'(def __init__\(self, source: str = None, show_popup: bool = True\):.*?)(# Event callbacks)'
    replacement1 = r'''\1# Session tracking for duration calculation
        self.session_start = datetime.now()
        
        \2'''
    
    new_content = re.sub(pattern1, replacement1, content, flags=re.DOTALL)
    if new_content != content:
        content = new_content
        changes_made = True
        print("[OK] Added session_start initialization")
    
    # Fix duration calculation in session log update
    pattern2 = r'(# Update recording duration\s+current_time = datetime\.now\(\)\s+if hasattr\(self, \'session_start\'\) and self\.session_start:\s+duration = \(current_time - self\.session_start\)\.total_seconds\(\)\s+session_data\["statistics"\]\["recording_duration_seconds"\] = int\(duration\)\s+else:\s+# Fallback: calculate from video start time if available\s+if hasattr\(self, \'video_start_time\'\) and self\.video_start_time:\s+duration = \(current_time - self\.video_start_time\)\.total_seconds\(\)\s+session_data\["statistics"\]\["recording_duration_seconds"\] = int\(duration\)\s+else:\s+session_data\["statistics"\]\["recording_duration_seconds"\] = 0)'
    
    replacement2 = r'''# Update recording duration
                current_time = datetime.now()
                if hasattr(self, 'session_start') and self.session_start:
                    duration = (current_time - self.session_start).total_seconds()
                    session_data["statistics"]["recording_duration_seconds"] = max(0, int(duration))
                else:
                    # Initialize session_start if missing
                    self.session_start = current_time
                    session_data["statistics"]["recording_duration_seconds"] = 0'''
    
    new_content = re.sub(pattern2, replacement2, content, flags=re.DOTALL)
    if new_content != content:
        content = new_content
        changes_made = True
        print("[OK] Fixed recording duration calculation")
    
    # Write the fixed content back if changes were made
    if changes_made:
        with open('video_processor.py', 'w', encoding='utf-8') as f:
            f.write(content)
        print("[OK] Recording duration fixes applied to video_processor.py")
        return True
    else:
        print("[INFO] No recording duration patterns found to fix")
        return False

if __name__ == "__main__":
    print("=== Fixing Array Ambiguity Errors and Recording Duration ===")
    
    array_success = fix_array_ambiguity_errors()
    duration_success = fix_recording_duration()
    
    if array_success or duration_success:
        print("[OK] Fixes applied successfully")
    else:
        print("[INFO] No changes made")
