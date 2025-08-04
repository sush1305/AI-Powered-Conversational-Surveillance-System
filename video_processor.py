import cv2
import numpy as np
import threading
import queue
import time
import os
from datetime import datetime
from typing import Optional, Callable, List
import logging
from collections import deque
from config import config

logger = logging.getLogger(__name__)

class VideoProcessor:
    """Handles real-time video capture and processing"""
    
    def __init__(self, source: str = None, show_popup: bool = True):
        self.source = source or config.video.source
        self.fps = config.video.fps
        self.resolution = config.video.resolution
        self.show_popup = show_popup
        self.buffer_seconds = config.video.record_buffer_seconds
        
        # Video capture
        self.cap = None
        self.is_running = False
        self.capture_thread = None
        self.frame_queue = queue.Queue(maxsize=10)
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # Video recording
        self.video_writer = None
        self.is_recording = False
        self.current_video_file = None
        self.video_start_time = None
        
        # Frame processing stats
        self.frame_count = 0
        self.current_fps = 0  # Rename to avoid overwriting config fps
        self.last_fps_time = time.time()
        
        # Create directories for storage
        self._create_storage_directories()
        
        # Frame buffer for event recording (will be resized when capture starts)
        self.frame_buffer = deque(maxlen=100)  # Default size, will be updated
        
        # Event callbacks
        self.event_callbacks: List[Callable] = []
        
        # Threading (will be started in start_capture)
        self.capture_thread = None
        
        # Popup display control (for main thread)
        self.popup_frame = None
        self.popup_frame_lock = threading.Lock()
        self.popup_window_created = False
        
    def add_event_callback(self, callback: Callable):
        """Add callback function for event detection"""
        self.event_callbacks.append(callback)
        
    def list_available_cameras(self, max_cameras=10) -> list:
        """Detect and return list of available camera indices"""
        available_cameras = []
        for i in range(max_cameras):
            cap = None
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap is not None and cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        available_cameras.append(i)
                        logger.info(f"Found camera at index {i}")
            except Exception as e:
                logger.debug(f"Camera check failed for index {i}: {e}")
            finally:
                if cap is not None:
                    cap.release()
        return available_cameras

    def start_capture(self) -> bool:
        """Start video capture with robust fallback options"""
        # Try multiple camera sources if the default fails
        sources_to_try = []
        
        if self.source.isdigit():
            # Try the specified camera index and fallbacks
            camera_index = int(self.source)
            sources_to_try = [camera_index, 0, 1, 2]  # Try specified, then common indices
        else:
            # For file paths or URLs, only try the specified source
            sources_to_try = [self.source]
            
        # Add dummy source as final fallback
        sources_to_try.append("dummy")
        
        for source in sources_to_try:
            try:
                logger.info(f"Trying video source: {source}")
                
                # Handle dummy source specially
                if source == "dummy":
                    logger.info("All camera sources failed, using dummy video source")
                    return self._create_dummy_source()
                
                # Initialize video capture with enhanced settings
                if isinstance(source, int):
                    self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)  # Use DirectShow on Windows
                else:
                    self.cap = cv2.VideoCapture(source)
                
                # Set buffer size to reduce latency
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Test if camera is accessible
                if not self.cap.isOpened():
                    logger.warning(f"Could not open video source: {source}")
                    if self.cap:
                        self.cap.release()
                    continue
                
                # Test reading multiple frames to ensure stability
                stable_reads = 0
                for _ in range(3):
                    ret, test_frame = self.cap.read()
                    if ret and test_frame is not None:
                        stable_reads += 1
                    time.sleep(0.1)
                
                if stable_reads < 2:
                    logger.warning(f"Unstable frame reading from source: {source} ({stable_reads}/3 successful)")
                    if self.cap:
                        self.cap.release()
                    continue
                
                # Success! Configure the camera
                logger.info(f"Successfully opened video source: {source}")
                
                # Set resolution (may not work for all cameras)
                try:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                    self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                except Exception as e:
                    logger.warning(f"Could not set camera properties: {e}")
                
                # Update source to the working one
                self.source = str(source)
                self.is_running = True
                
                # Update frame buffer size now that fps is set
                self.frame_buffer = deque(maxlen=self.fps * self.buffer_seconds)
                logger.info(f"Frame buffer sized for {self.fps * self.buffer_seconds} frames")
                
                # Start video recording
                self._start_video_recording()
                
                # Start the capture thread
                import threading
                self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
                self.capture_thread.start()
                
                logger.info(f"Video capture started successfully on source: {source}")
                logger.info(f"Capture thread started: {self.capture_thread.is_alive()}")
                return True
                
            except Exception as e:
                logger.warning(f"Error with video source {source}: {e}")
                if self.cap:
                    self.cap.release()
                continue
        
        # If all sources failed, create a dummy video source
        logger.error("All video sources failed. Creating dummy video source.")
        return self._create_dummy_source()
    
    def _create_dummy_source(self) -> bool:
        """Create a dummy video source when no camera is available"""
        try:
            logger.info("Creating dummy video source with synthetic frames")
            self.source = "dummy"
            self.is_running = True
            
            # Start dummy capture thread
            self.capture_thread = threading.Thread(target=self._dummy_capture_loop, daemon=True)
            self.capture_thread.start()
            
            # Start video recording
            self._start_video_recording()
            
            logger.info("Dummy video source started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create dummy video source: {e}")
            return False
    
    def _dummy_capture_loop(self):
        """Generate synthetic frames when no camera is available"""
        frame_time = 1.0 / self.fps
        frame_counter = 0
        
        # IMPORTANT: Stop any video recording in dummy mode
        if self.is_recording:
            self._stop_video_recording()
            logger.info("Stopped video recording in dummy mode")
        
        while self.is_running:
            start_time = time.time()
            
            # Create a synthetic frame
            frame = self._generate_dummy_frame(frame_counter)
            
            # Add timestamp to frame
            timestamp = datetime.now()
            timestamped_frame = {
                'frame': frame,
                'timestamp': timestamp,
                'frame_id': int(timestamp.timestamp() * 1000)
            }
            
            # Update current frame
            with self.frame_lock:
                self.current_frame = timestamped_frame
                
            
            # Update popup frame for display
            with self.popup_frame_lock:
                self.popup_frame = frame.copy()
                # Add to buffer
            self.frame_buffer.append(timestamped_frame)
            
            # Add to processing queue (non-blocking)
            try:
                self.frame_queue.put_nowait(timestamped_frame)
            except queue.Full:
                # Skip frame if queue is full
                pass
                
            # Process callbacks
            for callback in self.event_callbacks:
                try:
                    callback(timestamped_frame)
                except Exception as e:
                    logger.error(f"Error in event callback: {e}")
            
            frame_counter += 1
            
            # Maintain frame rate
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_time - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _generate_dummy_frame(self, frame_counter: int) -> np.ndarray:
        """Generate a synthetic frame for testing"""
        # Create a frame with the configured resolution
        height, width = self.resolution[1], self.resolution[0]
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add a gradient background
        for y in range(height):
            for x in range(width):
                frame[y, x] = [int(x * 255 / width), int(y * 255 / height), 128]
        
        # Add moving elements
        center_x = int(width / 2 + 100 * np.sin(frame_counter * 0.1))
        center_y = int(height / 2 + 50 * np.cos(frame_counter * 0.1))
        
        # Draw a moving circle
        cv2.circle(frame, (center_x, center_y), 30, (0, 255, 255), -1)
        
        # Add text overlay
        text = f"DEMO MODE - Frame {frame_counter}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add timestamp
        timestamp_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp_text, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
            
    def stop_capture(self):
        """Stop video capture"""
        self.is_running = False
        self.is_recording = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
            
        if self.cap:
            self.cap.release()
            
        # Close video writer
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            logger.info(f"Video recording stopped: {self.current_video_file}")
            
        logger.info("Video capture stopped")
        
    def _capture_loop(self):
        """Main capture loop running in separate thread"""
        frame_time = 1.0 / self.fps
        consecutive_failures = 0
        max_failures = 15  # Reduce failures before switching to dummy
        reinit_attempts = 0
        max_reinit_attempts = 3  # Try to reinitialize camera up to 3 times
        
        while self.is_running:
            start_time = time.time()
            
            ret, frame = self.cap.read()
            if not ret:
                consecutive_failures += 1
                logger.warning(f"Failed to read frame (attempt {consecutive_failures}/{max_failures})")
                
                # Try to reinitialize camera every 5 failures
                if consecutive_failures % 5 == 0 and consecutive_failures < max_failures and reinit_attempts < max_reinit_attempts:
                    reinit_attempts += 1
                    logger.info(f"Attempting to reinitialize camera (attempt {reinit_attempts}/{max_reinit_attempts})")
                    try:
                        if self.cap:
                            self.cap.release()
                        time.sleep(1.0)  # Wait before reinitializing
                        
                        # Reinitialize camera with enhanced settings
                        self.cap = cv2.VideoCapture(int(self.source), cv2.CAP_DSHOW)
                        if not self.cap.isOpened():
                            self.cap = cv2.VideoCapture(int(self.source))
                        
                        if self.cap.isOpened():
                            # Set camera properties
                            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                            
                            # Test if camera is working
                            test_ret, test_frame = self.cap.read()
                            if test_ret and test_frame is not None:
                                logger.info("Camera reinitialized successfully")
                                consecutive_failures = 0  # Reset counter
                                continue
                            else:
                                logger.warning("Camera reinitialization test failed")
                        else:
                            logger.warning("Camera reinitialization failed to open")
                    except Exception as e:
                        logger.error(f"Error reinitializing camera: {e}")
                
                # If too many failures, switch to dummy mode
                if consecutive_failures >= max_failures:
                    logger.error("Camera has failed consistently, switching to dummy video source")
                    
                    # Stop video recording to prevent dummy video files
                    if self.is_recording:
                        self._stop_video_recording()
                        logger.info("Video recording stopped due to camera failure")
                    
                    # Release current camera
                    if self.cap:
                        self.cap.release()
                        self.cap = None
                    
                    # Switch to dummy mode WITHOUT recording
                    self.source = "dummy"
                    logger.info("Starting dummy video capture loop (NO RECORDING)")
                    self._dummy_capture_loop()
                    return  # Exit this capture loop
                
                # Wait a bit before trying again
                time.sleep(0.1)
                continue
            
            # Reset failure counter on successful read
            consecutive_failures = 0
                
            # Add timestamp to frame
            timestamp = datetime.now()
            timestamped_frame = {
                'frame': frame,
                'timestamp': timestamp,
                'frame_id': int(timestamp.timestamp() * 1000)
            }
            
            # Update current frame
            with self.frame_lock:
                self.current_frame = timestamped_frame
                
            
            # Update popup frame for display
            with self.popup_frame_lock:
                self.popup_frame = frame.copy()
                # Add to buffer
            self.frame_buffer.append(timestamped_frame)
            
            # Add to processing queue (non-blocking)
            try:
                self.frame_queue.put_nowait(timestamped_frame)
            except queue.Full:
                # Skip frame if queue is full
                pass
                
            # Record frame to video file
            if self.is_recording:
                try:
                    if self.video_writer is None:
                        # Initialize video writer
                        height, width = frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        self.video_writer = cv2.VideoWriter(self.current_video_file, fourcc, self.fps, (width, height))
                        logger.info(f"Video writer initialized: {width}x{height} at {self.fps} fps")
                    
                    # Write frame to video
                    self.video_writer.write(frame)
                    
                except Exception as e:
                    logger.error(f"Error writing frame to video: {e}")
                
            # Display popup directly from capture thread (reliable approach)
            if self.show_popup:
                try:
                    display_frame = frame.copy()
                    
                    # Add overlay information with better visibility and updated controls
                    cv2.putText(display_frame, f"AI Surveillance System - {timestamp.strftime('%H:%M:%S')}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # Recording status indicator
                    recording_status = "RECORDING" if self.is_recording else "NOT RECORDING"
                    status_color = (0, 0, 255) if self.is_recording else (128, 128, 128)
                    cv2.putText(display_frame, recording_status, 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                    
                    # Control instructions
                    cv2.putText(display_frame, "Controls: 'q'=Stop Recording+Close | 'r'=Start Recording | ESC=Close Only", 
                               (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Add status indicator
                    cv2.circle(display_frame, (display_frame.shape[1] - 30, 30), 10, (0, 255, 0), -1)
                    cv2.putText(display_frame, "LIVE", (display_frame.shape[1] - 60, 35), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Popup display using the same approach as working standalone test
                    window_name = 'AI Surveillance - Live Feed'
                    
                    # Log popup display (only once)
                    if not hasattr(self, '_popup_logged'):
                        logger.info(f"[POPUP] Creating video popup window: {window_name}")
                        logger.info(f"[POPUP] Frame shape: {display_frame.shape}")
                        logger.info(f"[POPUP] OpenCV version: {cv2.__version__}")
                        self._popup_logged = True
                    
                    # Use the exact same approach as the working standalone test
                    try:
                        # Create window (same as standalone test)
                        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(window_name, 800, 600)
                        cv2.moveWindow(window_name, 100, 100)
                        
                        # Display frame (same as standalone test)
                        cv2.imshow(window_name, display_frame)
                        
                        # Log successful display (only once)
                        if not hasattr(self, '_popup_displayed'):
                            logger.info(f"[POPUP] Video popup window displayed successfully!")
                            self._popup_displayed = True
                        
                        # Check for 'q' key press with improved handling
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            logger.info("[POPUP] 'q' key pressed - stopping recording and closing popup")
                            self.show_popup = False
                            self.is_recording = False  # Stop recording when user presses 'q'
                            cv2.destroyAllWindows()
                            logger.info("[POPUP] Video popup closed and recording stopped by user")
                        elif key == ord('r'):  # 'r' key to restart recording
                            if not self.is_recording:
                                self._start_video_recording()
                                logger.info("[POPUP] Recording restarted by user ('r' key)")
                        elif key == 27:  # ESC key to close popup but keep recording
                            self.show_popup = False
                            cv2.destroyAllWindows()
                            logger.info("[POPUP] Video popup closed by ESC key - recording continues")
                            
                    except Exception as popup_error:
                        logger.error(f"[POPUP] Error in popup display: {popup_error}")
                        # Don't disable popup immediately, just log the error
                        pass
                        
                except Exception as e:
                    logger.error(f"Error displaying video popup: {e}")
                    logger.error(f"Frame shape: {frame.shape if frame is not None else 'None'}")
                    logger.error(f"OpenCV version: {cv2.__version__}")
                    import traceback
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    # Don't disable popup immediately, try a few more times
                    if not hasattr(self, '_popup_error_count'):
                        self._popup_error_count = 0
                    self._popup_error_count += 1
                    if self._popup_error_count > 10:
                        logger.warning("Too many popup errors, disabling popup display")
                        self.show_popup = False
            
            # Process callbacks
            for callback in self.event_callbacks:
                try:
                    callback(timestamped_frame)
                except Exception as e:
                    logger.error(f"Error in event callback: {e}")
                    
            # Maintain frame rate
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_time - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    def get_current_frame(self) -> Optional[dict]:
        """Get the current frame safely"""
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame else None
            
    def get_frame_for_processing(self, timeout: float = 1.0) -> Optional[dict]:
        """Get frame from processing queue"""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def get_recent_frames(self, seconds: int = 5) -> List[dict]:
        """Get recent frames from buffer"""
        if not self.frame_buffer:
            return []
            
        cutoff_time = datetime.now().timestamp() - seconds
        recent_frames = []
        
        for frame_data in reversed(self.frame_buffer):
            if frame_data['timestamp'].timestamp() >= cutoff_time:
                recent_frames.append(frame_data)
            else:
                break
                
        return list(reversed(recent_frames))
        
    def create_event_clip(self, event_timestamp: datetime, 
                         duration_seconds: int = None) -> List[dict]:
        """Create video clip around event timestamp"""
        duration = duration_seconds or config.video.event_clip_duration
        
        event_time = event_timestamp.timestamp()
        start_time = event_time - (duration // 2)
        end_time = event_time + (duration // 2)
        
        clip_frames = []
        for frame_data in self.frame_buffer:
            frame_time = frame_data['timestamp'].timestamp()
            if start_time <= frame_time <= end_time:
                clip_frames.append(frame_data)
                
        return clip_frames
        
    def save_clip_to_video(self, frames: List[dict], output_path: str) -> bool:
        """Save frames as video file"""
        if not frames:
            return False
            
        try:
            # Get frame dimensions
            height, width = frames[0]['frame'].shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
            
            for frame_data in frames:
                out.write(frame_data['frame'])
                
            out.release()
            logger.info(f"Saved video clip: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving video clip: {e}")
            return False
            
    def add_overlay_text(self, frame: np.ndarray, text: str, 
                        position: tuple = (10, 30), 
                        color: tuple = (0, 255, 0)) -> np.ndarray:
        """Add text overlay to frame"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # Add background rectangle for better readability
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(frame, 
                     (position[0] - 5, position[1] - text_height - 5),
                     (position[0] + text_width + 5, position[1] + 5),
                     (0, 0, 0), -1)
        
        # Add text
        cv2.putText(frame, text, position, font, font_scale, color, thickness)
        return frame
        
    def draw_bounding_box(self, frame: np.ndarray, bbox: tuple, 
                         label: str = "", confidence: float = 0.0,
                         color: tuple = (0, 255, 0)) -> np.ndarray:
        """Draw bounding box with label"""
        x1, y1, x2, y2 = bbox
        
        # Draw rectangle
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Draw label
        if label:
            label_text = f"{label}: {confidence:.2f}" if confidence > 0 else label
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            cv2.rectangle(frame,
                         (int(x1), int(y1) - label_size[1] - 10),
                         (int(x1) + label_size[0], int(y1)),
                         color, -1)
            
            cv2.putText(frame, label_text,
                       (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
        
    def get_frame_as_jpeg(self, frame: np.ndarray, quality: int = 90) -> bytes:
        """Convert frame to JPEG bytes"""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        return buffer.tobytes()
        
    def resize_frame(self, frame: np.ndarray, width: int = None, 
                    height: int = None) -> np.ndarray:
        """Resize frame maintaining aspect ratio"""
        if width is None and height is None:
            return frame
            
        h, w = frame.shape[:2]
        
        if width is None:
            # Calculate width based on height
            aspect_ratio = w / h
            width = int(height * aspect_ratio)
        elif height is None:
            # Calculate height based on width
            aspect_ratio = h / w
            height = int(width * aspect_ratio)
            
        return cv2.resize(frame, (width, height))
    
    def display_popup_main_thread(self) -> bool:
        """Display popup window from main thread (GUI-safe)"""
        if not self.show_popup:
            return False
        try:
            with self.popup_frame_lock:
                frame = self.popup_frame.copy() if self.popup_frame is not None else None
            if frame is None:
                # If no frame is available, show a black frame or wait
                frame = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
                cv2.putText(frame, "No video feed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            window_name = 'AI Surveillance - Live Feed'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            # Handle popup controls
            if key == ord('q'):
                self.stop_capture()
                cv2.destroyWindow(window_name)
                return False
            elif key == ord('r'):
                self._start_video_recording()
            elif key == 27:  # ESC
                cv2.destroyWindow(window_name)
                return False
            return True
        except Exception as e:
            logger.error(f"Error in popup display: {e}")
            return False
            
            # Log popup display attempt (only once)
            if not self.popup_window_created:
                logger.info(f"Creating popup window from main thread: {window_name}")
                logger.info(f"Frame shape: {display_frame.shape}")
                logger.info(f"OpenCV version: {cv2.__version__}")
                self.popup_window_created = True
            
            # Create window with specific properties
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(window_name, 800, 600)
            cv2.moveWindow(window_name, 100, 100)
            
            # Display the frame
            cv2.imshow(window_name, display_frame)
            
            # Force window update and bring to front
            try:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
            except:
                pass  # Some OpenCV versions don't support this
            
            # Check for 'q' key press to close popup (non-blocking)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.show_popup = False
                cv2.destroyAllWindows()
                logger.info("Video popup closed by user")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error displaying popup from main thread: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def _create_storage_directories(self):
        """Create enhanced storage directories with separate folders and logging"""
        import os
        from datetime import datetime
        
        # Create main storage directories with enhanced structure
        self.storage_dir = "surveillance_data"
        self.video_dir = os.path.join(self.storage_dir, "recorded_videos")
        self.harmful_snapshots_dir = os.path.join(self.storage_dir, "harmful_snapshots")
        self.session_logs_dir = os.path.join(self.storage_dir, "session_logs")
        self.harmful_events_log_dir = os.path.join(self.storage_dir, "harmful_events_log")
        self.database_dir = os.path.join(self.storage_dir, "database")
        
        # Create directories if they don't exist
        directories = [
            self.storage_dir, 
            self.video_dir, 
            self.harmful_snapshots_dir, 
            self.session_logs_dir,
            self.harmful_events_log_dir,
            self.database_dir
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
        logger.info(f"Enhanced storage structure created:")
        logger.info(f"  [VIDEOS] Recorded Videos: {self.video_dir}")
        logger.info(f"  [HARMFUL] Harmful Snapshots: {self.harmful_snapshots_dir}")
        logger.info(f"  [LOGS] Session Logs: {self.session_logs_dir}")
        logger.info(f"  [EVENTS] Harmful Events Log: {self.harmful_events_log_dir}")
        logger.info(f"  [DATABASE] Database: {self.database_dir}")
        
        # Create session-specific log files
        self._create_session_log_files()
    
    def _create_session_log_files(self):
        """Create session-specific log files in JSON and XLS format"""
        from datetime import datetime
        import os
        import json
        
        # Create session timestamp
        self.session_start = datetime.now()
        session_id = self.session_start.strftime('%Y-%m-%d_%H-%M-%S')
        
        # Session log files (JSON and XLS) - NEW FILE FOR EACH SESSION
        self.session_log_json = os.path.join(self.session_logs_dir, f"session_{session_id}.json")
        self.session_log_xls = os.path.join(self.session_logs_dir, f"session_{session_id}.xlsx")
        
        # Harmful events log files - SINGLE FILE FOR ALL SESSIONS
        self.harmful_events_json = os.path.join(self.harmful_events_log_dir, "harmful_events_log.json")
        self.harmful_events_xls = os.path.join(self.harmful_events_log_dir, "harmful_events_log.xlsx")
        
        # Initialize session log JSON
        session_data = {
            "session_info": {
                "session_id": session_id,
                "start_time": self.session_start.isoformat(),
                "system_version": "AI Surveillance System v2.0",
                "recording_status": "active"
            },
            "events": [],
            "statistics": {
                "total_events": 0,
                "harmful_events": 0,
                "recording_duration_seconds": 0
            }
        }
        
        with open(self.session_log_json, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2)
        
        # Initialize or load existing harmful events log structure
        if not os.path.exists(self.harmful_events_json):
            # Create new harmful events log file if it doesn't exist
            harmful_data = {
                "metadata": {
                    "created": self.session_start.isoformat(),
                    "last_updated": self.session_start.isoformat(),
                    "log_type": "harmful_events_only",
                    "description": "Single log file for all harmful events across all sessions"
                },
                "harmful_events": [],
                "summary": {
                    "total_harmful_events": 0,
                    "event_types": {},
                    "sessions_logged": []
                }
            }
            
            with open(self.harmful_events_json, 'w', encoding='utf-8') as f:
                json.dump(harmful_data, f, indent=2)
            
            logger.info(f"Created new harmful events log: {self.harmful_events_json}")
        else:
            # Update existing harmful events log with new session info
            try:
                with open(self.harmful_events_json, 'r', encoding='utf-8') as f:
                    harmful_data = json.load(f)
                
                # Update metadata
                harmful_data["metadata"]["last_updated"] = self.session_start.isoformat()
                if "sessions_logged" not in harmful_data["summary"]:
                    harmful_data["summary"]["sessions_logged"] = []
                
                harmful_data["summary"]["sessions_logged"].append({
                    "session_id": session_id,
                    "start_time": self.session_start.isoformat()
                })
                
                with open(self.harmful_events_json, 'w', encoding='utf-8') as f:
                    json.dump(harmful_data, f, indent=2)
                
                logger.info(f"Updated existing harmful events log: {self.harmful_events_json}")
            except Exception as e:
                logger.error(f"Error updating harmful events log: {e}")
                # Create new file if there's an error reading the existing one
                harmful_data = {
                    "metadata": {
                        "created": self.session_start.isoformat(),
                        "last_updated": self.session_start.isoformat(),
                        "log_type": "harmful_events_only",
                        "description": "Single log file for all harmful events across all sessions"
                    },
                    "harmful_events": [],
                    "summary": {
                        "total_harmful_events": 0,
                        "event_types": {},
                        "sessions_logged": [{
                            "session_id": session_id,
                            "start_time": self.session_start.isoformat()
                        }]
                    }
                }
                
                with open(self.harmful_events_json, 'w', encoding='utf-8') as f:
                    json.dump(harmful_data, f, indent=2)
        
        logger.info(f"Session log files created:")
        logger.info(f"  [SESSION] Session JSON: {self.session_log_json}")
        logger.info(f"  [HARMFUL] Harmful Events JSON: {self.harmful_events_json}")

# ... (rest of the code remains the same)
    def _start_video_recording(self):
        """Start continuous video recording"""
        try:
            import os
            from datetime import datetime
            
            # Create video filename with timestamp
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            self.current_video_file = os.path.join(self.video_dir, f"recording_{timestamp}.mp4")
            
            # Video recording will be handled in the capture loop
            self.is_recording = True
            self.video_start_time = datetime.now()
            
            logger.info(f"Video recording started: {self.current_video_file}")
            
        except Exception as e:
            logger.error(f"Error starting video recording: {e}")
            self.is_recording = False
    
    def _is_critical_event(self, event_type: str) -> bool:
        """Determine if event is critical enough to save snapshot"""
        critical_keywords = {
            'fall', 'violence', 'crash', 'robbery', 'weapon', 
            'accident', 'emergency', 'suspicious', 'fight',
            'attack', 'theft', 'break', 'danger', 'threat'
        }
        
        # Check if any critical keyword is in the event type
        event_lower = event_type.lower()
        return any(keyword in event_lower for keyword in critical_keywords)
    
    def _log_action(self, event_type: str, confidence: float, action_explanation: str, metadata: dict = None):
        """Log action to both JSON session log and harmful events log if applicable"""
        from datetime import datetime
        import json
        
        try:
            timestamp = datetime.now()
            
            # Create event entry
            event_entry = {
                "timestamp": timestamp.isoformat(),
                "event_type": event_type,
                "confidence": confidence,
                "action_explanation": action_explanation,
                "metadata": metadata or {},
                "is_harmful": self._is_critical_event(event_type)
            }
            
            # Add to session log
            self._append_to_session_log(event_entry)
            
            # Add to harmful events log if it's a critical event
            if self._is_critical_event(event_type):
                self._append_to_harmful_events_log(event_entry)
                
        except Exception as e:
            logger.error(f"Error logging action: {e}")
    
    def _append_to_session_log(self, event_entry: dict):
        """Append event to session log JSON file"""
        import json
        
        try:
            # Read current session log
            with open(self.session_log_json, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            # Add event
            session_data["events"].append(event_entry)
            session_data["statistics"]["total_events"] += 1
            
            if event_entry["is_harmful"]:
                session_data["statistics"]["harmful_events"] += 1
            
            # Update recording duration
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
                        session_data["statistics"]["recording_duration_seconds"] = 0
            
            # Write back to file
            with open(self.session_log_json, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error appending to session log: {e}")
    
    def _append_to_harmful_events_log(self, event_entry: dict):
        """Append harmful event to separate harmful events log"""
        import json
        from datetime import datetime
        
        try:
            # Read current harmful events log
            with open(self.harmful_events_json, 'r', encoding='utf-8') as f:
                harmful_data = json.load(f)
            
            # Add harmful event
            harmful_data["harmful_events"].append(event_entry)
            harmful_data["summary"]["total_harmful_events"] += 1
            
            # Update event type count
            event_type = event_entry["event_type"]
            if event_type in harmful_data["summary"]["event_types"]:
                harmful_data["summary"]["event_types"][event_type] += 1
            else:
                harmful_data["summary"]["event_types"][event_type] = 1
            
            # Write back to file
            with open(self.harmful_events_json, 'w', encoding='utf-8') as f:
                json.dump(harmful_data, f, indent=2)
                
            logger.info(f"Harmful event logged: {event_type} (confidence: {event_entry['confidence']:.2f})")
                
        except Exception as e:
            logger.error(f"Error appending to harmful events log: {e}")
    
    def save_critical_event_snapshot(self, frame: np.ndarray, event_type: str, 
                                   confidence: float, metadata: dict = None):
        """Save snapshot only for critical events with action logging"""
        if not self._is_critical_event(event_type):
            # Log non-critical event but don't save snapshot
            action_explanation = f"Non-critical event detected: {event_type}. No snapshot saved."
            self._log_action(event_type, confidence, action_explanation)
            return None
            
        try:
            from datetime import datetime
            import os
            
            timestamp = datetime.now()
            filename = f"snapshot_{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}_{event_type}.jpg"
            filepath = os.path.join(self.harmful_snapshots_dir, filename)
            
            # Save snapshot
            cv2.imwrite(filepath, frame)
            
            # Create metadata file
            metadata_file = filepath.replace('.jpg', '_metadata.json')
            event_metadata = {
                'timestamp': timestamp.isoformat(),
                'event_type': event_type,
                'confidence': confidence,
                'snapshot_file': filename,
                'frame_shape': frame.shape,
                'metadata': metadata or {}
            }
            
            import json
            with open(metadata_file, 'w') as f:
                json.dump(event_metadata, f, indent=2)
            
            # Log action with explanation
            action_explanation = f"Critical event detected! Snapshot saved: {filename}. Confidence: {confidence:.2f}. Metadata: {metadata or 'None'}"
            self._log_action(event_type, confidence, action_explanation)
            
            logger.info(f"Critical event snapshot saved: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving critical event snapshot: {e}")
            action_explanation = f"Failed to save critical event snapshot: {str(e)}"
            self._log_action(event_type, confidence, action_explanation)
            return None
    
    def _start_video_recording(self):
        """Start continuous video recording with robust codec selection"""
        try:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_video_file = os.path.join(self.video_dir, f"surveillance_{timestamp}.mp4")
            
            # Get frame dimensions from actual camera or default
            frame_width = self.resolution[0] if hasattr(self, 'resolution') else 640
            frame_height = self.resolution[1] if hasattr(self, 'resolution') else 480
            
            # Try multiple codecs for better compatibility
            codecs_to_try = [
                cv2.VideoWriter_fourcc(*'mp4v'),  # MPEG-4
                cv2.VideoWriter_fourcc(*'XVID'),  # XVID
                cv2.VideoWriter_fourcc(*'MJPG'),  # Motion JPEG
                cv2.VideoWriter_fourcc(*'X264')   # H.264
            ]
            
            self.video_writer = None
            
            for fourcc in codecs_to_try:
                try:
                    # Initialize video writer with current codec
                    test_writer = cv2.VideoWriter(
                        self.current_video_file,
                        fourcc,
                        self.fps,
                        (frame_width, frame_height)
                    )
                    
                    if test_writer.isOpened():
                        self.video_writer = test_writer
                        codec_name = {
                            cv2.VideoWriter_fourcc(*'mp4v'): 'MP4V',
                            cv2.VideoWriter_fourcc(*'XVID'): 'XVID', 
                            cv2.VideoWriter_fourcc(*'MJPG'): 'MJPG',
                            cv2.VideoWriter_fourcc(*'X264'): 'X264'
                        }.get(fourcc, 'Unknown')
                        logger.info(f"Video writer initialized with {codec_name} codec")
                        break
                    else:
                        test_writer.release()
                        
                except Exception as e:
                    logger.warning(f"Failed to initialize video writer with codec: {e}")
                    continue
            
            if self.video_writer and self.video_writer.isOpened():
                self.is_recording = True
                self.video_start_time = datetime.now()
                logger.info(f"Started recording video: {self.current_video_file}")
                logger.info(f"Video format: {frame_width}x{frame_height} at {self.fps} FPS")
            else:
                logger.error("Failed to initialize video writer with any codec")
                self.video_writer = None
                
        except Exception as e:
            logger.error(f"Error starting video recording: {e}")
    
    def _stop_video_recording(self):
        """Stop current video recording"""
        if self.video_writer and self.is_recording:
            try:
                self.video_writer.release()
                self.is_recording = False
                
                # Log recording info
                if self.video_start_time:
                    duration = datetime.now() - self.video_start_time
                    logger.info(f"Stopped recording: {self.current_video_file}")
                    logger.info(f"Recording duration: {duration}")
                    
                self.video_writer = None
                self.current_video_file = None
                self.video_start_time = None
                
            except Exception as e:
                logger.error(f"Error stopping video recording: {e}")
    
    def save_harmful_event(self, event_data: dict, frames: List[dict] = None):
        """Save harmful event with separate timestamp file"""
        try:
            import json
            import os
            
            # Create harmful event filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            event_file = os.path.join(self.harmful_events_dir, f"harmful_event_{timestamp}.json")
            
            # Add timestamp and additional info to event data
            event_data['timestamp'] = datetime.now().isoformat()
            event_data['video_file'] = self.current_video_file
            event_data['event_id'] = timestamp
            
            # Save event data
            with open(event_file, 'w') as f:
                json.dump(event_data, f, indent=2, default=str)
            
            # Save video clip if frames provided
            if frames:
                clip_file = os.path.join(self.harmful_events_dir, f"harmful_event_{timestamp}.mp4")
                self.save_clip_to_video(frames, clip_file)
                event_data['clip_file'] = clip_file
            
            logger.info(f"Harmful event saved: {event_file}")
            return event_file
            
        except Exception as e:
            logger.error(f"Error saving harmful event: {e}")
            return None
    
    def get_storage_info(self) -> dict:
        """Get information about storage locations and files"""
        import os
        import glob
        
        info = {
            'storage_directory': os.path.abspath(self.storage_dir),
            'video_directory': os.path.abspath(self.video_dir),
            'events_directory': os.path.abspath(self.events_dir),
            'harmful_events_directory': os.path.abspath(self.harmful_events_dir),
            'current_recording': self.current_video_file,
            'is_recording': self.is_recording
        }
        
        # Count files in each directory
        try:
            info['video_files_count'] = len(glob.glob(os.path.join(self.video_dir, '*.mp4')))
            info['event_files_count'] = len(glob.glob(os.path.join(self.events_dir, '*')))
            info['harmful_event_files_count'] = len(glob.glob(os.path.join(self.harmful_events_dir, '*')))
            
            # List recent files
            info['recent_videos'] = sorted(glob.glob(os.path.join(self.video_dir, '*.mp4')))[-5:]
            info['recent_harmful_events'] = sorted(glob.glob(os.path.join(self.harmful_events_dir, '*.json')))[-5:]
            
        except Exception as e:
            logger.warning(f"Error getting file counts: {e}")
            
        return info

class FrameAnalyzer:
    """Analyzes frames for basic properties and preprocessing"""
    
    @staticmethod
    def calculate_motion(frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate motion between two frames"""
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference
        diff = cv2.absdiff(gray1, gray2)
        
        # Calculate motion score
        motion_score = np.sum(diff) / (diff.shape[0] * diff.shape[1])
        return motion_score
        
    @staticmethod
    def detect_blur(frame: np.ndarray) -> float:
        """Detect blur in frame using Laplacian variance"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
        
    @staticmethod
    def adjust_brightness_contrast(frame: np.ndarray, 
                                  brightness: int = 0, 
                                  contrast: float = 1.0) -> np.ndarray:
        """Adjust brightness and contrast"""
        return cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
        
    @staticmethod
    def enhance_frame(frame: np.ndarray) -> np.ndarray:
        """Apply basic enhancement to frame"""
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return enhanced
