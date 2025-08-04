import cv2
import numpy as np
import os
import mediapipe as mp
import torch
from ultralytics import YOLO
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import threading
from collections import deque
from config import config
from database import db_manager, media_storage

logger = logging.getLogger(__name__)

class EventDetector:
    """Main event detection system combining multiple detection methods"""
    
    def __init__(self, video_processor=None):
        self.video_processor = video_processor
        
        # Increased confidence thresholds to reduce false positives
        self.base_confidence_threshold = max(config.detection.confidence_threshold, 0.7)  # Minimum 70%
        self.critical_confidence_threshold = 0.85  # 85% for critical events
        
        # Initialize detection modules with stricter settings
        self.fall_detector = FallDetector() if config.detection.fall_detection_enabled else None
        self.violence_detector = ViolenceDetector() if config.detection.violence_detection_enabled else None
        self.crash_detector = CrashDetector() if config.detection.vehicle_crash_detection_enabled else None
        self.fire_detector = FireDetector() if getattr(config.detection, 'fire_detection_enabled', True) else None
        self.weapon_detector = WeaponDetector() if getattr(config.detection, 'weapon_detection_enabled', True) else None
        self.intrusion_detector = IntrusionDetector() if getattr(config.detection, 'intrusion_detection_enabled', True) else None
        
        # Comprehensive critical event types for analytics
        self.critical_event_types = {
            'fall', 'falldown', 'violence', 'fight', 'vehicle_crash', 'crash', 
            'weapon_detected', 'gun', 'knife', 'fire', 'smoke', 'explosion',
            'robbery', 'theft', 'burglary', 'intrusion', 'trespassing',
            'emergency', 'accident', 'medical_emergency'
        }
        
        # Event buffer to prevent duplicate alerts
        self.recent_events = deque(maxlen=100)
        self.event_lock = threading.Lock()
        
        # Frame analysis for context
        self.frame_history = deque(maxlen=30)  # Keep last 30 frames for context
        self.motion_threshold = 50.0  # Minimum motion for event consideration
        
    def process_frame(self, frame_data: dict) -> List[Dict]:
        """Process frame and detect events"""
        frame = frame_data['frame']
        timestamp = frame_data['timestamp']
        events = []
        
        try:
            # Fall detection
            if self.fall_detector:
                fall_events = self.fall_detector.detect(frame, timestamp)
                events.extend(fall_events)
                
            # Violence detection
            if self.violence_detector:
                violence_events = self.violence_detector.detect(frame, timestamp)
                events.extend(violence_events)
                
            # Vehicle crash detection
            if self.crash_detector:
                crash_events = self.crash_detector.detect(frame, timestamp)
                events.extend(crash_events)
                
            # Fire detection
            if self.fire_detector:
                fire_events = self.fire_detector.detect(frame, timestamp)
                events.extend(fire_events)
                
            # Weapon detection
            if self.weapon_detector:
                weapon_events = self.weapon_detector.detect(frame, timestamp)
                events.extend(weapon_events)
                
            # Intrusion detection
            if self.intrusion_detector:
                intrusion_events = self.intrusion_detector.detect(frame, timestamp)
                events.extend(intrusion_events)
                
            # Apply strict filtering to reduce false positives
            validated_events = self._validate_events(events, frame)
            filtered_events = self._filter_duplicate_events(validated_events)
            
            # Ensure all events have proper structure
            processed_events = []
            for event in filtered_events:
                # Ensure event has event_type field (use type if missing)
                if 'event_type' not in event and 'type' in event:
                    event['event_type'] = event['type']
                elif 'event_type' not in event:
                    event['event_type'] = 'unknown'
                
                processed_events.append(event)
                self._save_event_and_snapshot(event, frame)
                
            return processed_events
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return []
    
    def _save_event_and_snapshot(self, event: Dict, frame: np.ndarray):
        """Save event data and snapshot if it's a harmful event"""
        try:
            # Check if this is a harmful event that needs a snapshot
            harmful_event_types = ['fall', 'violence', 'crash', 'fire', 'weapon', 'intrusion', 'medical', 'theft', 'explosion']
            event_type = event.get('event_type', event.get('type', 'unknown')).lower()
            
            if any(harmful_type in event_type for harmful_type in harmful_event_types):
                # Save snapshot image
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                snapshot_filename = f"{event_type}_{timestamp}.jpg"
                snapshot_path = os.path.join('surveillance_data', 'harmful_snapshots', snapshot_filename)
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
                
                # Save the frame as an image
                cv2.imwrite(snapshot_path, frame)
                
                # Add snapshot path to event metadata
                if 'metadata' not in event:
                    event['metadata'] = {}
                event['metadata']['snapshot_path'] = snapshot_path
                
                logger.info(f"Saved harmful event snapshot: {snapshot_path}")
                
        except Exception as e:
            logger.error(f"Error saving event snapshot: {e}")
    
    def _validate_events(self, events: List[Dict], frame: np.ndarray) -> List[Dict]:
        """Validate events to reduce false positives"""
        validated_events = []
        
        # Add current frame to history for context analysis
        self.frame_history.append(frame)
        
        for event in events:
            confidence = event.get('confidence', 0.0)
            event_type = event.get('type', 'unknown')
            
            # Apply minimum confidence threshold
            if confidence < self.base_confidence_threshold:
                logger.debug(f"Event {event_type} rejected: low confidence ({confidence:.2f} < {self.base_confidence_threshold})")
                continue
            
            # Additional validation based on event type
            validation_score = self._calculate_validation_score(event, frame)
            
            # Require higher validation score for critical events
            min_validation_score = 0.8 if event_type in self.critical_event_types else 0.6
            
            if validation_score >= min_validation_score:
                event['validation_score'] = validation_score
                event['confidence'] = min(confidence * validation_score, 1.0)  # Adjust confidence
                validated_events.append(event)
                logger.debug(f"Event {event_type} validated: confidence={confidence:.2f}, validation={validation_score:.2f}")
            else:
                logger.debug(f"Event {event_type} rejected: low validation score ({validation_score:.2f} < {min_validation_score})")
        
        return validated_events
    
    def _calculate_validation_score(self, event: Dict, frame: np.ndarray) -> float:
        """Calculate validation score based on context and motion analysis"""
        try:
            score = 1.0
            event_type = event.get('type', 'unknown')
            
            # Motion-based validation
            if len(self.frame_history) >= 2:
                motion_score = self._analyze_motion_context()
                
                # Different motion requirements for different events
                if event_type in ['fall', 'violence', 'vehicle_crash']:
                    # These events should have significant motion
                    if motion_score < self.motion_threshold:
                        score *= 0.3  # Heavily penalize low motion for motion-based events
                elif event_type in ['weapon_detected', 'suspicious_activity']:
                    # These events can have lower motion requirements
                    if motion_score < self.motion_threshold * 0.5:
                        score *= 0.7
            
            # Temporal consistency validation
            temporal_score = self._check_temporal_consistency(event)
            score *= temporal_score
            
            # Spatial validation (check if event location makes sense)
            spatial_score = self._validate_spatial_context(event, frame)
            score *= spatial_score
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating validation score: {e}")
            return 0.5  # Default moderate score on error
    
    def _analyze_motion_context(self) -> float:
        """Analyze motion between recent frames"""
        if len(self.frame_history) < 2:
            return 0.0
        
        try:
            current_frame = self.frame_history[-1]
            previous_frame = self.frame_history[-2]
            
            # Convert to grayscale
            gray1 = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate frame difference
            diff = cv2.absdiff(gray1, gray2)
            motion_pixels = np.sum(diff > 30)  # Threshold for significant change
            total_pixels = diff.shape[0] * diff.shape[1]
            
            motion_ratio = motion_pixels / total_pixels
            motion_score = motion_ratio * 1000  # Scale up for easier threshold comparison
            
            return motion_score
            
        except Exception as e:
            logger.error(f"Error analyzing motion context: {e}")
            return 0.0
    
    def _check_temporal_consistency(self, event: Dict) -> float:
        """Check if event is temporally consistent with recent events"""
        try:
            event_type = event.get('type', 'unknown')
            current_time = datetime.now().timestamp()
            
            # Check for similar recent events
            similar_recent_count = 0
            for recent_event in list(self.recent_events)[-10:]:  # Check last 10 events
                if (recent_event.get('type') == event_type and 
                    current_time - recent_event.get('timestamp', 0) < 5.0):  # Within 5 seconds
                    similar_recent_count += 1
            
            # Penalize if too many similar events recently (likely false positives)
            if similar_recent_count > 2:
                return 0.3  # Heavily penalize
            elif similar_recent_count > 1:
                return 0.7  # Moderately penalize
            else:
                return 1.0  # No penalty
                
        except Exception as e:
            logger.error(f"Error checking temporal consistency: {e}")
            return 1.0
    
    def _validate_spatial_context(self, event: Dict, frame: np.ndarray) -> float:
        """Validate spatial context of the event"""
        try:
            bbox = event.get('bbox')
            if not bbox:
                return 1.0  # No spatial info to validate
            
            # Check if bounding box is reasonable
            x1, y1, x2, y2 = bbox
            frame_height, frame_width = frame.shape[:2]
            
            # Check if bbox is within frame bounds
            if (x1 < 0 or y1 < 0 or x2 > frame_width or y2 > frame_height or
                x1 >= x2 or y1 >= y2):
                return 0.2  # Invalid bbox
            
            # Check bbox size (too small or too large might be false positive)
            bbox_area = (x2 - x1) * (y2 - y1)
            frame_area = frame_width * frame_height
            area_ratio = bbox_area / frame_area
            
            if area_ratio < 0.001:  # Too small (< 0.1% of frame)
                return 0.4
            elif area_ratio > 0.8:  # Too large (> 80% of frame)
                return 0.5
            else:
                return 1.0  # Good size
                
        except Exception as e:
            logger.error(f"Error validating spatial context: {e}")
            return 1.0
            
    def _filter_duplicate_events(self, events: List[Dict]) -> List[Dict]:
        """Filter out duplicate events within time window"""
        filtered = []
        current_time = datetime.now().timestamp()
        
        with self.event_lock:
            for event in events:
                # Check if similar event occurred recently (stricter duplicate detection)
                is_duplicate = False
                current_time = datetime.now().timestamp()
                event['timestamp'] = current_time  # Add timestamp to event
                
                for recent_event in self.recent_events:
                    if (current_time - recent_event['timestamp'] < 5.0 and  # 5 second window
                        recent_event['type'] == event['type'] and
                        self._calculate_bbox_overlap(recent_event.get('bbox'), event.get('bbox')) > 0.5):
                        is_duplicate = True
                        break
                        
                if not is_duplicate:
                    event['timestamp'] = current_time
                    self.recent_events.append(event)
                    filtered.append(event)
                    
        return filtered
        
    def _calculate_bbox_overlap(self, bbox1: Optional[Tuple], bbox2: Optional[Tuple]) -> float:
        """Calculate overlap between two bounding boxes"""
        if not bbox1 or not bbox2:
            return 0.0
            
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
            
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
        
    def _save_event(self, event: Dict, frame: np.ndarray):
        """Save event to database with media"""
        try:
            # Save frame as image
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"event_{event['type']}_{timestamp_str}.jpg"
            
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            media_data = buffer.tobytes()
            
            # Save media
            media_path = media_storage.save_media(media_data, filename)
            
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(v) for v in obj]
                return obj
            
            # Convert event data to JSON-serializable format
            serializable_metadata = convert_numpy_types(event.get('metadata', {}))
            confidence_score = float(event['confidence']) if isinstance(event['confidence'], np.floating) else event['confidence']
            
            # Save to database
            event_type = event['type']
            description = event['description']
            confidence = event['confidence']
            metadata = event.get('metadata', {})
            event_id = db_manager.add_event(
                event_type=event_type,
                description=description,
                confidence=float(confidence),
                metadata=json.dumps(metadata, default=str),
                media_path=media_path
            )
            
            # Check if this is a harmful event that needs separate storage
            harmful_events = ['fall_detected', 'violence_detected', 'crash_detected', 'robbery_detected']
            if event_type in harmful_events:
                self._save_harmful_event_separately(event_type, description, confidence, metadata, media_path)
            
            logger.info(f"Saved event {event_id}: {event_type}")
            
        except Exception as e:
            logger.error(f"Error saving event: {e}")
            
    def _save_harmful_event_separately(self, event_type, description, confidence, metadata, media_path):
        """Save harmful event to separate file with timestamp"""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{event_type}_{timestamp_str}.txt"
        event_dir = "harmful_events"
        
        if not os.path.exists(event_dir):
            os.makedirs(event_dir)
        
        with open(os.path.join(event_dir, filename), "w") as f:
            f.write(f"Event Type: {event_type}\n")
            f.write(f"Description: {description}\n")
            f.write(f"Confidence: {confidence}\n")
            f.write(f"Metadata: {json.dumps(metadata, default=str)}\n")
            f.write(f"Media Path: {media_path}\n")
            
        logger.info(f"Saved harmful event to separate file: {filename}")

class FallDetector:
    """Detects person falls using MediaPipe pose estimation"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def detect(self, frame: np.ndarray, timestamp: datetime) -> List[Dict]:
        """Detect falls in frame"""
        events = []
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # Analyze pose for fall detection
                landmarks = results.pose_landmarks.landmark
                fall_confidence = self._analyze_pose_for_fall(landmarks)
                
                if fall_confidence > 0.7:
                    # Get bounding box of person
                    bbox = self._get_person_bbox(landmarks, frame.shape)
                    
                    events.append({
                        'type': 'fall',
                        'description': f'Person fall detected with confidence {fall_confidence:.2f}',
                        'confidence': fall_confidence,
                        'bbox': bbox,
                        'metadata': {
                            'detection_method': 'mediapipe_pose',
                            'pose_landmarks': self._landmarks_to_dict(landmarks)
                        }
                    })
                    
        except Exception as e:
            logger.error(f"Error in fall detection: {e}")
            
        return events
        
    def _analyze_pose_for_fall(self, landmarks):
        """Analyze pose landmarks to detect fall with stricter criteria"""
        try:
            # Get key body landmarks
            nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
            left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
            right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
            left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
            right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]
            
            # Calculate body orientation with more precision
            shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
            hip_center_y = (left_hip.y + right_hip.y) / 2
            ankle_center_y = (left_ankle.y + right_ankle.y) / 2
            knee_center_y = (left_knee.y + right_knee.y) / 2
            
            # Check landmark visibility (MediaPipe confidence)
            min_visibility = 0.7
            key_landmarks = [nose, left_shoulder, right_shoulder, left_hip, right_hip, left_ankle, right_ankle]
            if any(lm.visibility < min_visibility for lm in key_landmarks):
                return False  # Not enough visible landmarks
            
            # More strict fall detection criteria
            body_height = abs(nose.y - ankle_center_y)
            body_width = abs(left_shoulder.x - right_shoulder.x)
            
            # Fall indicators with stricter thresholds
            aspect_ratio = body_width / (body_height + 0.001)
            is_horizontal = aspect_ratio > 2.0  # More strict: body must be significantly wider than tall
            is_very_low = shoulder_center_y > 0.8  # Shoulders must be very low in frame
            is_head_low = nose.y > hip_center_y  # Head lower than hips (strong fall indicator)
            
            # Check if knees are at unusual position (not bent normally)
            knee_hip_distance = abs(knee_center_y - hip_center_y)
            is_legs_extended = knee_hip_distance < 0.1  # Legs not properly bent
            
            # Calculate confidence with stricter requirements
            confidence = 0.0
            if is_horizontal:
                confidence += 0.3
            if is_very_low:
                confidence += 0.3
            if is_head_low:
                confidence += 0.4  # Strong indicator
            if is_legs_extended:
                confidence += 0.2
                
            # Require much higher confidence (85%) to reduce false positives
            return confidence > 0.85
            
        except Exception as e:
            logger.error(f"Error analyzing pose for fall: {e}")
            return False
            
    def _get_person_bbox(self, landmarks, frame_shape) -> Tuple[int, int, int, int]:
        """Get bounding box of detected person"""
        h, w = frame_shape[:2]
        
        x_coords = [lm.x * w for lm in landmarks]
        y_coords = [lm.y * h for lm in landmarks]
        
        x1 = max(0, int(min(x_coords)) - 20)
        y1 = max(0, int(min(y_coords)) - 20)
        x2 = min(w, int(max(x_coords)) + 20)
        y2 = min(h, int(max(y_coords)) + 20)
        
        return (x1, y1, x2, y2)
        
    def _landmarks_to_dict(self, landmarks) -> Dict:
        """Convert landmarks to dictionary for storage"""
        return {
            i: {'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility}
            for i, lm in enumerate(landmarks)
        }

class ViolenceDetector:
    """Detects violent behavior using YOLO and motion analysis"""
    
    def __init__(self):
        self.confidence_threshold = config.detection.confidence_threshold
        try:
            self.model = YOLO(config.detection.yolo_model_path)
            self.violence_classes = ['person', 'knife', 'gun']  # Classes that might indicate violence
            self.motion_threshold = 50.0
            self.previous_frame = None
            
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            self.model = None
            
    def detect(self, frame: np.ndarray, timestamp: datetime) -> List[Dict]:
        """Detect violent behavior in frame"""
        events = []
        
        if not self.model:
            return events
            
        try:
            # Run YOLO detection with enhanced error handling
            try:
                results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            except AttributeError as attr_err:
                if "'Conv' object has no attribute 'bn'" in str(attr_err):
                    logger.warning("YOLO model compatibility issue detected, skipping violence detection for this frame")
                    return events
                else:
                    raise attr_err
            
            # Analyze detections for violence indicators
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = self.model.names[class_id]
                        
                        # Check for weapon detection
                        if class_name in ['knife', 'gun'] and confidence > 0.6:
                            bbox = box.xyxy[0].cpu().numpy()
                            
                            events.append({
                                'type': 'weapon_detected',
                                'description': f'{class_name.title()} detected with confidence {confidence:.2f}',
                                'confidence': confidence,
                                'bbox': tuple(bbox.astype(int)),
                                'metadata': {
                                    'detection_method': 'yolo',
                                    'weapon_type': class_name
                                }
                            })
                            
            # Analyze motion for violent behavior
            if self.previous_frame is not None:
                motion_events = self._analyze_motion_for_violence(frame, self.previous_frame)
                events.extend(motion_events)
                
            self.previous_frame = frame.copy()
            
        except Exception as e:
            logger.error(f"Error in violence detection: {e}")
            
        return events
        
    def _analyze_motion_for_violence(self, current_frame: np.ndarray, previous_frame: np.ndarray) -> List[Dict]:
        """Analyze motion patterns for violent behavior with stricter criteria"""
        events = []
        
        try:
            # Convert to grayscale
            gray1 = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate frame difference with higher threshold
            diff = cv2.absdiff(gray1, gray2)
            
            # Use higher threshold to filter out minor movements
            motion_threshold = 50  # Increased from 30
            motion_mask = diff > motion_threshold
            
            # Calculate motion metrics
            motion_pixels = np.sum(motion_mask)
            total_pixels = diff.shape[0] * diff.shape[1]
            motion_ratio = motion_pixels / total_pixels
            
            # Calculate intensity of motion (only for significant motion)
            if motion_pixels > 0:
                motion_intensity = np.mean(diff[motion_mask])
                # Normalize intensity
                intensity_score = min(motion_intensity / 255.0, 1.0)
            else:
                intensity_score = 0.0
            
            # Calculate motion distribution (violence often has scattered motion)
            motion_regions = cv2.connectedComponents(motion_mask.astype(np.uint8))[0]
            region_score = min(motion_regions / 20.0, 1.0)  # More regions = higher score
            
            # Combine metrics with stricter weighting
            base_score = (motion_ratio * 0.4 + intensity_score * 0.4 + region_score * 0.2)
            
            # Apply much stricter scaling - require very high motion for violence
            violence_score = max(0.0, (base_score - 0.3) * 2.0)  # Subtract baseline, then scale

            # Only return an event if violence_score is high enough
            if violence_score > 0.75:
                events.append({
                    'type': 'violence',
                    'description': 'Violent motion detected (motion analysis)',
                    'confidence': float(violence_score),
                    'metadata': {
                        'motion_ratio': float(motion_ratio),
                        'intensity_score': float(intensity_score),
                        'region_score': float(region_score),
                    }
                })
            return events
        except Exception as e:
            logger.error(f"Error analyzing motion for violence: {e}")
            return []
    
    def _check_person_consistency(self) -> bool:
        """Check if person detections are consistent over recent frames"""
        if len(self.person_history) < 3:
            return False
        
        # Check if we consistently detect multiple people
        recent_counts = [len(detections) for detections in list(self.person_history)[-3:]]
        return all(count >= 2 for count in recent_counts)
    
    def _check_temporal_consistency(self, event: Dict) -> float:
        """Check if event is temporally consistent with recent events"""
        try:
            event_type = event.get('type', 'unknown')
            current_time = datetime.now().timestamp()
            
            # Check for similar recent events
            similar_recent_count = 0
            for recent_event in list(self.recent_events)[-10:]:  # Check last 10 events
                if (recent_event.get('type') == event_type and 
                    current_time - recent_event.get('timestamp', 0) < 5.0):  # Within 5 seconds
                    similar_recent_count += 1
            
            # Penalize if too many similar events recently (likely false positives)
            if similar_recent_count > 2:
                return 0.3  # Heavily penalize
            elif similar_recent_count > 1:
                return 0.7  # Moderately penalize
            else:
                return 1.0  # No penalty
                
        except Exception as e:
            logger.error(f"Error checking temporal consistency: {e}")
            return 1.0

class CrashDetector:
    """Detects vehicle crashes using YOLO and motion analysis"""
    
    def __init__(self):
        # Much higher confidence threshold for violence detection to reduce false positives
        self.confidence_threshold = max(config.detection.confidence_threshold, 0.8)  # Minimum 80%
        self.violence_confidence_threshold = 0.9  # 90% for violence events
        
        try:
            self.model = YOLO('yolov8n.pt')
            logger.info("Violence detection YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading YOLO model for violence detection: {e}")
            self.model = None
        
        self.previous_frame = None
        self.previous_detections = []  # Track previous detections for crash analysis
        self.motion_history = deque(maxlen=10)  # Track motion over time
        self.person_history = deque(maxlen=5)   # Track person detections
        
    def detect(self, frame: np.ndarray, timestamp: datetime) -> List[Dict]:
        """Detect vehicle crashes in frame"""
        events = []
        
        if not self.model:
            return events
            
        try:
            # Run YOLO detection with enhanced error handling
            try:
                results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            except AttributeError as attr_err:
                if "'Conv' object has no attribute 'bn'" in str(attr_err):
                    logger.warning("YOLO model compatibility issue detected, skipping crash detection for this frame")
                    return events
                else:
                    raise attr_err
            current_detections = []
            
            # Extract person detections with stricter filtering
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = self.model.names[class_id]
                        
                        # Only consider high-confidence person detections
                        if class_name == 'person' and confidence > self.confidence_threshold:
                            bbox = box.xyxy[0].cpu().numpy()
                            
                            # Additional filtering: check bbox size (avoid tiny detections)
                            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                            frame_area = frame.shape[0] * frame.shape[1]
                            area_ratio = bbox_area / frame_area
                            
                            # Only consider reasonably sized person detections
                            if 0.01 < area_ratio < 0.7:  # Between 1% and 70% of frame
                                current_detections.append({
                                    'bbox': bbox,
                                    'confidence': confidence,
                                    'area_ratio': area_ratio
                                })
            
            # Analyze for crashes
            if self.previous_detections:
                crash_events = self._analyze_for_crashes(current_detections, self.previous_detections)
                events.extend(crash_events)
                
            self.previous_detections = current_detections
            
            # Store current detections for history tracking
            self.person_history.append(current_detections)
            
            # Analyze motion for violence if we have previous frame
            if self.previous_frame is not None:
                motion_score = self._analyze_motion_for_violence(frame, self.previous_frame)
                self.motion_history.append(motion_score)
                
                # Much stricter violence detection criteria
                avg_motion = sum(self.motion_history) / len(self.motion_history) if self.motion_history else 0
                person_count = len(current_detections)
                
                # Require multiple strict conditions for violence detection
                violence_indicators = {
                    'high_motion': motion_score > 0.7,  # Very high motion threshold
                    'sustained_motion': avg_motion > 0.5,  # Sustained high motion
                    'multiple_people': person_count >= 2,  # At least 2 people
                    'consistent_people': self._check_person_consistency(),  # Consistent person detection
                }
                
                # Count how many indicators are met (fix float/numpy.float64 iterable error)
                indicators_met = sum([1 for k, v in violence_indicators.items() if isinstance(v, bool) and v])
                
                # Require at least 3 out of 4 indicators for violence detection
                if indicators_met >= 3:
                    # Calculate final confidence based on multiple factors
                    base_confidence = motion_score * 0.4 + avg_motion * 0.3 + (person_count / 5.0) * 0.3
                    final_confidence = min(base_confidence * (indicators_met / 4.0), 1.0)
                    
                    # Only report if confidence exceeds very high threshold
                    if final_confidence >= self.violence_confidence_threshold:
                        events.append({
                            'type': 'violence',
                            'description': f'High-confidence violent behavior detected',
                            'confidence': final_confidence,
                            'bbox': current_detections[0]['bbox'] if current_detections else None,
                            'metadata': {
                                'detection_method': 'enhanced_motion_analysis',
                                'motion_score': motion_score,
                                'avg_motion': avg_motion,
                                'person_count': person_count,
                                'indicators_met': indicators_met,
                                'indicators': violence_indicators
                            }
                        })
                        logger.info(f"Violence detected with {indicators_met}/4 indicators, confidence: {final_confidence:.2f}")
            
            self.previous_frame = frame.copy()
            
        except Exception as e:
            logger.error(f"Error in crash detection: {e}")
            
        return events
    
    def _analyze_motion_for_violence(self, current_frame: np.ndarray, previous_frame: np.ndarray) -> float:
        """Analyze motion between frames for violence detection"""
        try:
            # Convert frames to grayscale
            gray1 = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            
            # Use simpler motion detection for synthetic frames
            # Calculate frame difference instead of optical flow for dummy frames
            diff = cv2.absdiff(gray1, gray2)
            motion_pixels = np.sum(diff > 30)  # Count pixels with significant change
            total_pixels = diff.shape[0] * diff.shape[1]
            motion_ratio = motion_pixels / total_pixels
            
            # Normalize motion ratio to 0-1 scale
            return min(motion_ratio * 10.0, 1.0)  # Amplify small changes
                
        except Exception as e:
            logger.error(f"Error analyzing motion for violence: {e}")
            return 0.0
    
    def _check_person_consistency(self) -> bool:
        """Check if person detections are consistent over time"""
        try:
            if len(self.person_history) < 3:
                return False
            
            # Check if we have consistent person detections
            recent_counts = [len(detections) for detections in list(self.person_history)[-3:]]
            avg_count = sum(recent_counts) / len(recent_counts)
            
            # Consider consistent if average is >= 2 people
            return avg_count >= 2.0
            
        except Exception as e:
            logger.error(f"Error checking person consistency: {e}")
            return False
        
    def _analyze_for_crashes(self, current: List[Dict], previous: List[Dict]) -> List[Dict]:
        """Analyze vehicle movements for crash detection"""
        events = []
        
        try:
            # Look for sudden stops or collisions
            for curr_vehicle in current:
                for prev_vehicle in previous:
                    # Match vehicles by proximity and class
                    curr_center = curr_vehicle['center']
                    prev_center = prev_vehicle['center']
                    
                    distance = np.sqrt((curr_center[0] - prev_center[0])**2 + 
                                     (curr_center[1] - prev_center[1])**2)
                    
                    if (distance < 50 and  # Same vehicle (close proximity)
                        curr_vehicle['class'] == prev_vehicle['class']):
                        
                        # Check for sudden movement changes
                        movement = np.sqrt((curr_center[0] - prev_center[0])**2 + 
                                       (curr_center[1] - prev_center[1])**2)
                        
                        if movement > 100:  # Sudden large movement
                            confidence = min(movement / 200.0, 1.0)
                            
                            events.append({
                                'type': 'vehicle_crash',
                                'description': f'Potential {curr_vehicle["class"]} crash detected',
                                'confidence': confidence,
                                'bbox': tuple(curr_vehicle['bbox'].astype(int)),
                                'metadata': {
                                    'detection_method': 'movement_analysis',
                                    'vehicle_type': curr_vehicle['class'],
                                    'movement_distance': movement
                                }
                            })
                            
        except Exception as e:
            logger.error(f"Error analyzing crashes: {e}")
            
        return events

class FireDetector:
    """Detects fire and smoke using color analysis and YOLO"""
    
    def __init__(self):
        self.confidence_threshold = max(config.detection.confidence_threshold, 0.8)
        self.fire_confidence_threshold = 0.85  # High threshold for fire detection
        
        try:
            self.model = YOLO('yolov8n.pt')
            logger.info("Fire detection YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading YOLO model for fire detection: {e}")
            self.model = None
        
        # Fire/smoke color ranges in HSV
        self.fire_lower = np.array([0, 50, 50])    # Lower bound for fire colors
        self.fire_upper = np.array([35, 255, 255]) # Upper bound for fire colors
        self.smoke_lower = np.array([0, 0, 50])    # Lower bound for smoke colors
        self.smoke_upper = np.array([180, 50, 200]) # Upper bound for smoke colors
        
    def detect(self, frame: np.ndarray, timestamp: datetime) -> List[Dict]:
        """Detect fire and smoke in frame"""
        events = []
        
        try:
            # Color-based fire detection
            fire_detected, fire_confidence, fire_bbox = self._detect_fire_by_color(frame)
            if fire_detected:
                events.append({
                    'type': 'fire',
                    'description': f'Fire detected using color analysis',
                    'confidence': fire_confidence,
                    'bbox': fire_bbox,
                    'metadata': {
                        'detection_method': 'color_analysis',
                        'fire_area_ratio': fire_confidence
                    }
                })
            
            # Smoke detection
            smoke_detected, smoke_confidence, smoke_bbox = self._detect_smoke_by_color(frame)
            if smoke_detected:
                events.append({
                    'type': 'smoke',
                    'description': f'Smoke detected using color analysis',
                    'confidence': smoke_confidence,
                    'bbox': smoke_bbox,
                    'metadata': {
                        'detection_method': 'color_analysis',
                        'smoke_area_ratio': smoke_confidence
                    }
                })
            
        except Exception as e:
            logger.error(f"Error in fire detection: {e}")
            
        return events
    
    def _detect_fire_by_color(self, frame: np.ndarray) -> Tuple[bool, float, Optional[Tuple]]:
        """Detect fire using color analysis"""
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Create mask for fire colors
            fire_mask = cv2.inRange(hsv, self.fire_lower, self.fire_upper)
            
            # Apply morphological operations to reduce noise
            kernel = np.ones((5, 5), np.uint8)
            fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
            fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel)
            
            # Calculate fire area ratio
            fire_pixels = np.sum(fire_mask > 0)
            total_pixels = frame.shape[0] * frame.shape[1]
            fire_ratio = fire_pixels / total_pixels
            
            # Fire detection threshold
            if fire_ratio > 0.02:  # At least 2% of frame should be fire-colored
                # Find bounding box of fire region
                contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    bbox = (x, y, x + w, y + h)
                    
                    confidence = min(fire_ratio * 20, 1.0)  # Scale confidence
                    return confidence >= self.fire_confidence_threshold, confidence, bbox
            
            return False, 0.0, None
            
        except Exception as e:
            logger.error(f"Error in fire color detection: {e}")
            return False, 0.0, None
    
    def _detect_smoke_by_color(self, frame: np.ndarray) -> Tuple[bool, float, Optional[Tuple]]:
        """Detect smoke using color analysis"""
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Create mask for smoke colors (grayish)
            smoke_mask = cv2.inRange(hsv, self.smoke_lower, self.smoke_upper)
            
            # Apply morphological operations
            kernel = np.ones((7, 7), np.uint8)
            smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN, kernel)
            
            # Calculate smoke area ratio
            smoke_pixels = np.sum(smoke_mask > 0)
            total_pixels = frame.shape[0] * frame.shape[1]
            smoke_ratio = smoke_pixels / total_pixels
            
            # Smoke detection threshold (higher than fire since smoke can be more common)
            if smoke_ratio > 0.05:  # At least 5% of frame should be smoke-colored
                # Find bounding box
                contours, _ = cv2.findContours(smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    bbox = (x, y, x + w, y + h)
                    
                    confidence = min(smoke_ratio * 15, 1.0)  # Scale confidence
                    return confidence >= 0.7, confidence, bbox  # Lower threshold for smoke
            
            return False, 0.0, None
            
        except Exception as e:
            logger.error(f"Error in smoke color detection: {e}")
            return False, 0.0, None

class WeaponDetector:
    """Detects weapons using YOLO object detection"""
    
    def __init__(self):
        self.confidence_threshold = max(config.detection.confidence_threshold, 0.85)
        self.weapon_confidence_threshold = 0.9  # Very high threshold for weapons
        
        try:
            self.model = YOLO('yolov8n.pt')
            logger.info("Weapon detection YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading YOLO model for weapon detection: {e}")
            self.model = None
        
        # Weapon-related classes (if available in YOLO model)
        self.weapon_classes = ['knife', 'gun', 'rifle', 'pistol', 'weapon']
        
    def detect(self, frame: np.ndarray, timestamp: datetime) -> List[Dict]:
        """Detect weapons in frame"""
        events = []
        
        if not self.model:
            return events
        
        try:
            # Run YOLO detection with very high confidence threshold
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = self.model.names[class_id]
                        
                        # Check if detected object is weapon-related
                        if any(weapon in class_name.lower() for weapon in self.weapon_classes):
                            if confidence >= self.weapon_confidence_threshold:
                                bbox = box.xyxy[0].cpu().numpy()
                                
                                events.append({
                                    'type': 'weapon_detected',
                                    'description': f'{class_name.title()} detected with high confidence',
                                    'confidence': confidence,
                                    'bbox': tuple(bbox.astype(int)),
                                    'metadata': {
                                        'detection_method': 'yolo_object_detection',
                                        'weapon_type': class_name,
                                        'severity': 'critical'
                                    }
                                })
                                
                                logger.warning(f"WEAPON DETECTED: {class_name} with confidence {confidence:.2f}")
            
        except Exception as e:
            logger.error(f"Error in weapon detection: {e}")
            
        return events

class IntrusionDetector:
    """Detects unauthorized intrusion and trespassing"""
    
    def __init__(self):
        self.confidence_threshold = max(config.detection.confidence_threshold, 0.75)
        self.intrusion_confidence_threshold = 0.8
        
        try:
            self.model = YOLO('yolov8n.pt')
            logger.info("Intrusion detection YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading YOLO model for intrusion detection: {e}")
            self.model = None
        
        # Track person history for intrusion analysis
        self.person_history = deque(maxlen=30)
        self.restricted_zones = []  # Can be configured for specific areas
        
    def detect(self, frame: np.ndarray, timestamp: datetime) -> List[Dict]:
        """Detect intrusion and trespassing"""
        events = []
        
        if not self.model:
            return events
        
        try:
            # Detect persons in frame
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            current_persons = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = self.model.names[class_id]
                        
                        if class_name == 'person' and confidence > self.confidence_threshold:
                            bbox = box.xyxy[0].cpu().numpy()
                            current_persons.append({
                                'bbox': bbox,
                                'confidence': confidence,
                                'timestamp': timestamp
                            })
            
            # Store current detections
            self.person_history.append(current_persons)
            
            # Analyze for suspicious intrusion patterns
            intrusion_events = self._analyze_intrusion_patterns(current_persons, frame)
            events.extend(intrusion_events)
            
        except Exception as e:
            logger.error(f"Error in intrusion detection: {e}")
            
        return events
    
    def _analyze_intrusion_patterns(self, current_persons: List[Dict], frame: np.ndarray) -> List[Dict]:
        """Analyze patterns that might indicate intrusion"""
        events = []
        
        try:
            if not current_persons:
                return events
            
            # Check for multiple people (potential group intrusion)
            if len(current_persons) >= 3:
                avg_confidence = sum(p['confidence'] for p in current_persons) / len(current_persons)
                
                if avg_confidence >= self.intrusion_confidence_threshold:
                    # Get bounding box that encompasses all persons
                    all_bboxes = [p['bbox'] for p in current_persons]
                    min_x = min(bbox[0] for bbox in all_bboxes)
                    min_y = min(bbox[1] for bbox in all_bboxes)
                    max_x = max(bbox[2] for bbox in all_bboxes)
                    max_y = max(bbox[3] for bbox in all_bboxes)
                    
                    events.append({
                        'type': 'intrusion',
                        'description': f'Multiple persons detected - potential group intrusion ({len(current_persons)} people)',
                        'confidence': avg_confidence,
                        'bbox': (int(min_x), int(min_y), int(max_x), int(max_y)),
                        'metadata': {
                            'detection_method': 'group_analysis',
                            'person_count': len(current_persons),
                            'severity': 'high' if len(current_persons) >= 4 else 'medium'
                        }
                    })
            
            # Check for persistent presence (loitering)
            if len(self.person_history) >= 10:  # At least 10 frames of history
                recent_frames_with_people = sum(1 for frame_persons in list(self.person_history)[-10:] if len(frame_persons) > 0)
                
                if recent_frames_with_people >= 8:  # People present in 8 out of 10 recent frames
                    events.append({
                        'type': 'trespassing',
                        'description': 'Persistent unauthorized presence detected',
                        'confidence': 0.8,
                        'bbox': current_persons[0]['bbox'] if current_persons else None,
                        'metadata': {
                            'detection_method': 'persistence_analysis',
                            'frames_with_people': recent_frames_with_people,
                            'severity': 'medium'
                        }
                    })
            
        except Exception as e:
            logger.error(f"Error analyzing intrusion patterns: {e}")
            
        return events
