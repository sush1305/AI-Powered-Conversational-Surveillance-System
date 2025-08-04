#!/usr/bin/env python3
"""
AI-Powered Conversational Surveillance System
Main application entry point
"""

import os
import sys
import logging
import asyncio
import signal
import threading
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import config
from database import db_manager, media_storage
from video_processor import VideoProcessor
from event_detector import EventDetector
from nlp_interface import nlp_interface
from voice_interface import voice_interface
from notification_system import notification_manager
from web_dashboard import app
import uvicorn

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.system.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.system.log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class SurveillanceSystem:
    """Main surveillance system orchestrator"""
    
    def __init__(self):
        self.video_processor = None
        self.event_detector = None
        self.is_running = False
        self.processing_thread = None
        
    def initialize(self):
        """Initialize all system components"""
        logger.info("Initializing AI Surveillance System...")
        
        try:
            # Create required directories
            self._create_directories()
            
            # Validate configuration
            config_issues = config.validate()
            if config_issues:
                logger.warning("Configuration issues found:")
                for issue in config_issues:
                    logger.warning(f"  - {issue}")
            
            # Initialize database
            logger.info("Initializing database...")
            db_manager.init_database()
            
            # Initialize video processor with popup enabled
            logger.info("Initializing video processor...")
            self.video_processor = VideoProcessor(show_popup=True)
            
            # Initialize event detector
            logger.info("Initializing event detector...")
            self.event_detector = EventDetector()
            
            # Set up event processing callback
            self.video_processor.add_event_callback(self._process_frame_for_events)
            
            # Initialize NLP interface with fallback
            logger.info("Initializing NLP interface...")
            try:
                from nlp_interface import NLPInterface
                self.nlp_interface = NLPInterface()
                if not self.nlp_interface.llm_available:
                    logger.info("LLM not available, using simple NLP interface...")
                    from simple_nlp_interface import SimpleNLPInterface
                    self.nlp_interface = SimpleNLPInterface()
                else:
                    # Only update semantic search if using full NLP interface
                    self.nlp_interface.semantic_search.update_from_database()
            except Exception as e:
                logger.warning(f"Standard NLP interface failed, using simple fallback: {e}")
                from simple_nlp_interface import SimpleNLPInterface
                self.nlp_interface = SimpleNLPInterface()
            
            # Initialize voice interface
            logger.info("Initializing voice interface...")
            voice_interface.set_query_callback(self._handle_voice_query)
            
            # Test notification system
            logger.info("Testing notification system...")
            notification_results = notification_manager.test_notifications()
            logger.info(f"Notification test results: {notification_results}")
            
            logger.info("System initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return False
            
    def _create_directories(self):
        """Create required directories"""
        directories = [
            config.database.local_media_path,
            config.ai.vector_db_path,
            "templates",
            "static",
            "logs"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
    def start(self):
        """Start the surveillance system"""
        if not self.initialize():
            logger.error("Failed to initialize system")
            return False
            
        logger.info("Starting surveillance system...")
        
        try:
            # Start video capture
            if self.video_processor.start_capture():
                logger.info("Video capture started")
            else:
                logger.error("Failed to start video capture")
                return False
                
            # Start processing thread
            self.is_running = True
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
            
            # Start voice interface if configured
            if voice_interface.microphone:
                voice_interface.start_listening()
                logger.info("Voice interface started")
            
            # Send startup notification
            notification_manager.send_system_notification(
                "System Started",
                "AI Surveillance System has started successfully",
                "INFO"
            )
            
            logger.info("Surveillance system started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start surveillance system: {e}")
            return False
            
    def stop(self):
        """Stop the surveillance system"""
        logger.info("Stopping surveillance system...")
        
        try:
            # Stop processing
            self.is_running = False
            
            # Stop video capture
            if self.video_processor:
                self.video_processor.stop_capture()
                
            # Stop voice interface
            voice_interface.stop_listening()
            
            # Wait for processing thread to finish
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=5.0)
                
            # Send shutdown notification
            notification_manager.send_system_notification(
                "System Stopped",
                "AI Surveillance System has been stopped",
                "INFO"
            )
            
            logger.info("Surveillance system stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping surveillance system: {e}")
            
    def _processing_loop(self):
        """Main processing loop for video frames"""
        logger.info("Starting video processing loop...")
        
        while self.is_running:
            try:
                # Get frame from video processor
                frame_data = self.video_processor.get_frame_for_processing(timeout=1.0)
                
                if frame_data:
                    # Process frame for events
                    self._process_frame_for_events(frame_data)
                    
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                
        logger.info("Video processing loop stopped")
        
    def _process_frame_for_events(self, frame_data):
        """Process frame for event detection"""
        try:
            if self.event_detector:
                events = self.event_detector.process_frame(frame_data)
                
                for event in events:
                    logger.info(f"Event detected: {event['type']} - {event['description']}")
                    
                    # Send notifications
                    notification_manager.send_alert(event)
                    
                    # Add to semantic search index
                    event_id = event.get('event_id')
                    if event_id:
                        nlp_interface.semantic_search.add_event_embedding(
                            event_id, event['description']
                        )
                        
        except Exception as e:
            logger.error(f"Error processing frame for events: {e}")
            
    def _handle_voice_query(self, query: str, result: dict):
        """Handle voice query results"""
        try:
            logger.info(f"Voice query processed: {query}")
            
            # Log query result
            db_manager.log_system_event(
                "INFO",
                f"Voice query: {query} - Found {result.get('total_count', 0)} results",
                "voice_interface"
            )
            
        except Exception as e:
            logger.error(f"Error handling voice query: {e}")
            
    def get_system_status(self):
        """Get current system status"""
        return {
            'running': self.is_running,
            'video_capture': self.video_processor.is_running if self.video_processor else False,
            'event_detection': self.event_detector is not None,
            'voice_interface': voice_interface.is_listening,
            'database': os.path.exists(config.database.sqlite_path),
            'notifications': any([
                notification_manager.sms_client,
                notification_manager.email_config,
                notification_manager.pushbullet_client
            ])
        }

# Global system instance
surveillance_system = SurveillanceSystem()

def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown"""
    logger.info(f"Received signal {signum}, shutting down...")
    surveillance_system.stop()
    sys.exit(0)

def main():
    """Main entry point"""
    print("=" * 60)
    print("AI-Powered Conversational Surveillance System")
    print("=" * 60)
    print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start surveillance system
        if not surveillance_system.start():
            logger.error("Failed to start surveillance system")
            sys.exit(1)
            
        # Start web dashboard
        logger.info(f"Starting web dashboard on {config.web.host}:{config.web.port}")
        
        # Display prominent web link and video popup info
        print("\n" + "=" * 80)
        print("🎥 AI SURVEILLANCE SYSTEM - READY TO USE!")
        print("=" * 80)
        print(f"📱 WEB DASHBOARD: http://{config.web.host}:{config.web.port}")
        print("   ↳ Access live video, events, analytics, and AI chat")
        print("\n[VIDEO] VIDEO POPUP: Live video feed window will open automatically")
        print("   [->] Press 'q' in the video window to STOP RECORDING and close popup")
        print("   [->] Press 'r' in the video window to START/RESTART recording")
        print("   [->] Press ESC in the video window to close popup but keep recording")
        print("   [->] System continues running even if popup is closed")
        print("\n[STORAGE] FILE LOCATIONS:")
        print("   [->] All recordings: surveillance_data/videos/")
        print("   [->] Harmful events: surveillance_data/harmful_events/")
        print("   [->] Regular events: surveillance_data/events/")
        print("\n[VOICE] VOICE COMMANDS: Say 'surveillance' + your question")
        print("   [->] Example: 'surveillance, what events happened today?'")
        print("\n[CTRL] CONTROLS:")
        print("   [->] Press Ctrl+C to stop the entire system")
        print("   [->] Close video popup with 'q' (system keeps running)")
        print("=" * 80)
        print()
        
        # Start web server in a background thread so main thread can run popup
        import threading
        def run_web():
            uvicorn.run(
                app,
                host=config.web.host,
                port=config.web.port,
                log_level=config.system.log_level.lower(),
                access_log=True
            )
        web_thread = threading.Thread(target=run_web, daemon=True)
        web_thread.start()

        # Main-thread popup loop (required for OpenCV on Windows)
        vp = getattr(surveillance_system, 'video_processor', None)
        if vp and getattr(vp, 'show_popup', False):
            import time
            while surveillance_system.is_running and vp.show_popup:
                vp.display_popup_main_thread()
                time.sleep(0.01)
        # Wait for web server to finish
        web_thread.join()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        surveillance_system.stop()
        print("\nSystem shutdown complete")

if __name__ == "__main__":
    main()
