#!/usr/bin/env python3
"""
Storage Information Utility for AI Surveillance System
Shows where all your recorded files and events are stored
"""

import os
import glob
import json
from datetime import datetime

def check_storage_locations():
    """Check and display all storage locations and file counts"""
    
    print("=" * 80)
    print("[STORAGE] AI SURVEILLANCE SYSTEM - STORAGE INFORMATION")
    print("=" * 80)
    
    # Define storage directories
    base_dir = "surveillance_data"
    video_dir = os.path.join(base_dir, "videos")
    events_dir = os.path.join(base_dir, "events")
    harmful_events_dir = os.path.join(base_dir, "harmful_events")
    
    # Check if directories exist
    directories = {
        "Main Storage": base_dir,
        "Video Recordings": video_dir,
        "Regular Events": events_dir,
        "Harmful Events": harmful_events_dir
    }
    
    print("\n[DIRECTORIES] DIRECTORY LOCATIONS:")
    print("-" * 50)
    
    for name, path in directories.items():
        abs_path = os.path.abspath(path)
        exists = "[OK] EXISTS" if os.path.exists(path) else "[X] NOT FOUND"
        print(f"{name:20}: {abs_path}")
        print(f"{'':20}  Status: {exists}")
        print()
    
    # Count files in each directory
    print("[FILES] FILE COUNTS:")
    print("-" * 50)
    
    try:
        # Video files
        video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
        print(f"Video Recordings: {len(video_files)} files")
        
        if video_files:
            print("   Recent videos:")
            for video in sorted(video_files)[-3:]:
                filename = os.path.basename(video)
                size_mb = os.path.getsize(video) / (1024 * 1024)
                print(f"   - {filename} ({size_mb:.1f} MB)")
        
        print()
        
        # Harmful event files
        harmful_json_files = glob.glob(os.path.join(harmful_events_dir, "*.json"))
        harmful_txt_files = glob.glob(os.path.join(harmful_events_dir, "*.txt"))
        harmful_video_files = glob.glob(os.path.join(harmful_events_dir, "*.mp4"))
        
        print(f"Harmful Events: {len(harmful_json_files + harmful_txt_files)} event files")
        print(f"Harmful Event Videos: {len(harmful_video_files)} video clips")
        
        if harmful_json_files or harmful_txt_files:
            print("   Recent harmful events:")
            all_harmful = sorted(harmful_json_files + harmful_txt_files)[-3:]
            for event_file in all_harmful:
                filename = os.path.basename(event_file)
                print(f"   - {filename}")
        
        print()
        
        # Regular events (database entries)
        db_file = "surveillance.db"
        if os.path.exists(db_file):
            print(f"Database Events: surveillance.db exists")
            size_kb = os.path.getsize(db_file) / 1024
            print(f"   Database size: {size_kb:.1f} KB")
        else:
            print("Database Events: No database file found")
        
    except Exception as e:
        print(f"Error counting files: {e}")
    
    print("\n[ACCESS] HOW TO ACCESS YOUR FILES:")
    print("-" * 50)
    print("1. Open File Explorer")
    print(f"2. Navigate to: {os.path.abspath(base_dir)}")
    print("3. You'll find three folders:")
    print("   - videos/         : All continuous recordings")
    print("   - events/         : Regular event data")
    print("   - harmful_events/ : Critical events (falls, violence, crashes)")
    
    print("\n[WEB] WEB DASHBOARD ACCESS:")
    print("-" * 50)
    print("Open your browser and go to: http://127.0.0.1:8000")
    print("(Only when the surveillance system is running)")
    
    print("\n[COMMANDS] QUICK COMMANDS:")
    print("-" * 50)
    print("- Run surveillance: python main.py")
    print("- Check storage:    python check_storage.py")
    print("- Test video popup: python test_video_popup.py")
    
    print("=" * 80)

def show_recent_harmful_events():
    """Show details of recent harmful events"""
    harmful_events_dir = os.path.join("surveillance_data", "harmful_events")
    
    if not os.path.exists(harmful_events_dir):
        print("No harmful events directory found.")
        return
    
    # Find recent harmful event files
    json_files = glob.glob(os.path.join(harmful_events_dir, "*.json"))
    txt_files = glob.glob(os.path.join(harmful_events_dir, "*.txt"))
    
    all_files = sorted(json_files + txt_files, key=os.path.getmtime, reverse=True)
    
    if not all_files:
        print("No harmful events recorded yet.")
        return
    
    print("\n[HARMFUL] RECENT HARMFUL EVENTS:")
    print("-" * 50)
    
    for i, event_file in enumerate(all_files[:5]):  # Show last 5 events
        filename = os.path.basename(event_file)
        mod_time = datetime.fromtimestamp(os.path.getmtime(event_file))
        
        print(f"{i+1}. {filename}")
        print(f"   Time: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Try to read event details
        try:
            if event_file.endswith('.json'):
                with open(event_file, 'r') as f:
                    data = json.load(f)
                    print(f"   Type: {data.get('event_type', 'Unknown')}")
                    print(f"   Description: {data.get('description', 'N/A')}")
                    print(f"   Confidence: {data.get('confidence', 'N/A')}")
            elif event_file.endswith('.txt'):
                with open(event_file, 'r') as f:
                    lines = f.readlines()[:3]  # First 3 lines
                    for line in lines:
                        print(f"   {line.strip()}")
        except Exception as e:
            print(f"   Error reading file: {e}")
        
        print()

if __name__ == "__main__":
    check_storage_locations()
    show_recent_harmful_events()
