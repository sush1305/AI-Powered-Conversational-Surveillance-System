from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
import uvicorn
import cv2
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import base64
import os
from pathlib import Path

from config import config
from database import db_manager, media_storage
from nlp_interface import nlp_interface
from voice_interface import voice_interface
from notification_system import notification_manager
from video_processor import VideoProcessor
from event_detector import EventDetector

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="AI Surveillance System", version="1.0.0")

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/media", StaticFiles(directory=config.database.local_media_path), name="media")

# Global instances
video_processor = None
event_detector = None
connected_websockets: List[WebSocket] = []

class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
        
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove disconnected connections
                self.active_connections.remove(connection)

manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    """Initialize system components on startup"""
    global video_processor, event_detector
    
    try:
        # Initialize video processor
        video_processor = VideoProcessor()
        
        # Initialize event detector
        event_detector = EventDetector()
        
        # Add event callback to video processor
        video_processor.add_event_callback(process_frame_for_events)
        
        # Start video capture
        if video_processor.start_capture():
            logger.info("Video capture started successfully")
        else:
            logger.error("Failed to start video capture")
            
        # Initialize voice interface callback
        voice_interface.set_query_callback(handle_voice_query)
        
        logger.info("Surveillance system started successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global video_processor
    
    if video_processor:
        video_processor.stop_capture()
        
    voice_interface.stop_listening()
    logger.info("Surveillance system shutdown complete")

def process_frame_for_events(frame_data: dict):
    """Process frame for event detection and broadcast to clients"""
    try:
        if event_detector:
            events = event_detector.process_frame(frame_data)
            
            if events:
                # Broadcast events to connected clients (skip for now to avoid asyncio issues)
                try:
                    # TODO: Implement proper WebSocket broadcasting when needed
                    logger.debug(f"Events detected for broadcasting: {len(events)}")
                except Exception as e:
                    logger.debug(f"Event broadcasting skipped: {e}")
                
                # Send notifications
                for event in events:
                    notification_manager.send_alert(event)
                    
    except Exception as e:
        logger.error(f"Error processing frame for events: {e}")

async def broadcast_events(events: List[Dict]):
    """Broadcast events to all connected WebSocket clients"""
    for event in events:
        message = {
            'type': 'event',
            'data': {
                'event_type': event['type'],
                'description': event['description'],
                'confidence': event['confidence'],
                'timestamp': datetime.now().isoformat(),
                'bbox': event.get('bbox')
            }
        }
        await manager.broadcast(json.dumps(message))

def handle_voice_query(query: str, result: dict):
    """Handle voice query results"""
    try:
        # Broadcast query result to connected clients
        message = {
            'type': 'voice_query',
            'data': {
                'query': query,
                'result': result,
                'timestamp': datetime.now().isoformat()
            }
        }
        asyncio.create_task(manager.broadcast(json.dumps(message)))
        
    except Exception as e:
        logger.error(f"Error handling voice query: {e}")

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page"""
    # Get recent events for display
    recent_events = db_manager.get_events(limit=10)
    
    # Get system status
    system_status = {
        'video_capture': video_processor.is_running if video_processor else False,
        'event_detection': event_detector is not None,
        'voice_interface': voice_interface.microphone is not None,
        'notifications': any([
            notification_manager.sms_client,
            notification_manager.email_config,
            notification_manager.pushbullet_client
        ])
    }
    
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "recent_events": recent_events,
        "system_status": system_status
    })

@app.get("/events")
async def get_events(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    event_type: Optional[str] = None,
    limit: int = 100
):
    """Get events with optional filtering"""
    try:
        start_time = None
        end_time = None
        
        if start_date:
            start_time = datetime.fromisoformat(start_date)
        if end_date:
            end_time = datetime.fromisoformat(end_date)
            
        events = db_manager.get_events(
            start_time=start_time,
            end_time=end_time,
            event_type=event_type,
            limit=limit
        )
        
        # Convert datetime objects to strings for JSON serialization
        for event in events:
            if isinstance(event['timestamp'], datetime):
                event['timestamp'] = event['timestamp'].isoformat()
            if isinstance(event['created_at'], datetime):
                event['created_at'] = event['created_at'].isoformat()
                
        return JSONResponse(content={"events": events})
        
    except Exception as e:
        logger.error(f"Error getting events: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def process_query(request: Request):
    """Process natural language query"""
    try:
        data = await request.json()
        query = data.get('query', '')
        
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
            
        result = nlp_interface.process_query(query)
        
        # Convert datetime objects for JSON serialization
        for event in result.get('results', []):
            if isinstance(event.get('timestamp'), datetime):
                event['timestamp'] = event['timestamp'].isoformat()
            if isinstance(event.get('created_at'), datetime):
                event['created_at'] = event['created_at'].isoformat()
                
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/video_feed")
def video_feed():
    """Get current video frame as JPEG image"""
    from fastapi.responses import Response
    
    try:
        # Get current frame
        frame_data = None
        if video_processor and video_processor.is_running:
            frame_data = video_processor.get_current_frame()
        
        if frame_data and frame_data.get('frame') is not None:
            # Process real frame
            frame = frame_data['frame']
            timestamp = frame_data['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
            
            # Add overlay
            display_frame = frame.copy()
            display_frame = video_processor.add_overlay_text(display_frame, timestamp)
            
            # Encode as JPEG
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
            success, buffer = cv2.imencode('.jpg', display_frame, encode_params)
            
            if success:
                frame_bytes = buffer.tobytes()
            else:
                # Fallback to placeholder
                placeholder = create_placeholder_frame()
                _, buffer = cv2.imencode('.jpg', placeholder)
                frame_bytes = buffer.tobytes()
        else:
            # Use placeholder frame
            placeholder = create_placeholder_frame()
            _, buffer = cv2.imencode('.jpg', placeholder)
            frame_bytes = buffer.tobytes()
        
        # Return single JPEG frame
        return Response(
            content=frame_bytes,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
        
    except Exception as e:
        logger.error(f"Video feed error: {e}")
        # Return placeholder on error
        try:
            placeholder = create_placeholder_frame()
            _, buffer = cv2.imencode('.jpg', placeholder)
            frame_bytes = buffer.tobytes()
            return Response(
                content=frame_bytes,
                media_type="image/jpeg",
                headers={"Cache-Control": "no-cache"}
            )
        except:
            # Return minimal error response
            return Response(
                content=b"Error generating frame",
                media_type="text/plain",
                status_code=500
            )

def create_placeholder_frame():
    """Create placeholder frame when video is not available"""
    import numpy as np
    
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "No Video Feed", (200, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get('type') == 'ping':
                await websocket.send_text(json.dumps({'type': 'pong'}))
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/system_status")
async def get_system_status():
    """Get current system status"""
    try:
        status = {
            'video_capture': {
                'running': video_processor.is_running if video_processor else False,
                'source': config.video.source,
                'fps': config.video.fps
            },
            'event_detection': {
                'enabled': event_detector is not None,
                'fall_detection': config.detection.fall_detection_enabled,
                'violence_detection': config.detection.violence_detection_enabled,
                'crash_detection': config.detection.vehicle_crash_detection_enabled
            },
            'voice_interface': {
                'available': voice_interface.microphone is not None,
                'listening': voice_interface.is_listening,
                'enabled': voice_interface.voice_enabled
            },
            'notifications': {
                'sms': notification_manager.sms_client is not None,
                'email': notification_manager.email_config is not None,
                'push': notification_manager.pushbullet_client is not None
            },
            'database': {
                'path': config.database.sqlite_path,
                'media_storage': config.database.media_storage_type
            }
        }
        
        return JSONResponse(content=status)
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/voice/start")
async def start_voice_interface():
    """Start voice interface"""
    try:
        success = voice_interface.start_listening()
        return JSONResponse(content={"success": success})
    except Exception as e:
        logger.error(f"Error starting voice interface: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/voice/stop")
async def stop_voice_interface():
    """Stop voice interface"""
    try:
        voice_interface.stop_listening()
        return JSONResponse(content={"success": True})
    except Exception as e:
        logger.error(f"Error stopping voice interface: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test_notifications")
async def test_notifications():
    """Test all notification channels"""
    try:
        results = notification_manager.test_notifications()
        return JSONResponse(content=results)
    except Exception as e:
        logger.error(f"Error testing notifications: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/statistics")
async def get_statistics():
    """Get comprehensive system analytics"""
    try:
        # Get all events from database and harmful events log
        events = db_manager.get_events(limit=5000)  # Increased limit for better analytics
        
        # Also read from harmful events log for complete picture
        harmful_events = []
        harmful_log_path = Path(config.database.local_media_path) / "harmful_events_log" / "harmful_events_log.json"
        if harmful_log_path.exists():
            try:
                with open(harmful_log_path, 'r') as f:
                    harmful_data = json.load(f)
                    harmful_events = harmful_data.get('events', [])
            except Exception as e:
                logger.warning(f"Could not read harmful events log: {e}")
        
        # Combine all events for comprehensive analysis
        all_events = events + harmful_events
        
        # Initialize comprehensive event type tracking
        event_type_mapping = {
            # Safety events
            'fall': ['fall', 'falldown', 'falling'],
            'crash': ['crash', 'vehicle_crash', 'accident', 'collision'],
            'fire': ['fire', 'flame', 'burning'],
            'fight': ['fight', 'violence', 'fighting', 'assault'],
            'weapon': ['weapon_detected', 'gun', 'knife', 'rifle', 'pistol'],
            'intrusion': ['intrusion', 'trespassing', 'unauthorized_entry'],
            'smoke': ['smoke', 'smoking'],
            'medical': ['medical_emergency', 'health_emergency'],
            'theft': ['theft', 'robbery', 'burglary', 'stealing'],
            'explosion': ['explosion', 'blast']
        }
        
        # Count events by normalized categories
        normalized_counts = {category: 0 for category in event_type_mapping.keys()}
        raw_event_counts = {}
        severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        
        for event in all_events:
            event_type = event.get('event_type', event.get('type', 'unknown')).lower()
            raw_event_counts[event_type] = raw_event_counts.get(event_type, 0) + 1
            
            # Map to normalized categories
            for category, keywords in event_type_mapping.items():
                if any(keyword in event_type for keyword in keywords):
                    normalized_counts[category] += 1
                    break
            
            # Count by severity
            confidence = event.get('confidence', 0)
            metadata = event.get('metadata', {})
            severity = metadata.get('severity', 'low')
            
            if severity in severity_counts:
                severity_counts[severity] += 1
            elif confidence >= 0.9:
                severity_counts['critical'] += 1
            elif confidence >= 0.8:
                severity_counts['high'] += 1
            elif confidence >= 0.7:
                severity_counts['medium'] += 1
            else:
                severity_counts['low'] += 1
        
        # Time-based analysis
        end_date = datetime.now()
        start_date_week = end_date - timedelta(days=7)
        start_date_month = end_date - timedelta(days=30)
        
        # Daily events for last week
        daily_events = {}
        for i in range(7):
            date = start_date_week + timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")
            daily_events[date_str] = 0
        
        # Hourly distribution
        hourly_distribution = {str(i): 0 for i in range(24)}
        
        # Weekly events by category
        weekly_by_category = {category: 0 for category in event_type_mapping.keys()}
        monthly_by_category = {category: 0 for category in event_type_mapping.keys()}
        
        for event in all_events:
            # Parse timestamp
            event_timestamp = event.get('timestamp')
            if isinstance(event_timestamp, str):
                try:
                    event_date = datetime.fromisoformat(event_timestamp.replace('Z', '+00:00'))
                except:
                    continue
            elif isinstance(event_timestamp, datetime):
                event_date = event_timestamp
            else:
                continue
            
            # Daily events (last week)
            if start_date_week <= event_date <= end_date:
                date_str = event_date.strftime("%Y-%m-%d")
                if date_str in daily_events:
                    daily_events[date_str] += 1
            
            # Hourly distribution
            hour = str(event_date.hour)
            hourly_distribution[hour] += 1
            
            # Weekly and monthly category analysis
            event_type = event.get('event_type', event.get('type', 'unknown')).lower()
            for category, keywords in event_type_mapping.items():
                if any(keyword in event_type for keyword in keywords):
                    if start_date_week <= event_date <= end_date:
                        weekly_by_category[category] += 1
                    if start_date_month <= event_date <= end_date:
                        monthly_by_category[category] += 1
                    break
        
        # Calculate trends (week over week)
        trends = {}
        for category in event_type_mapping.keys():
            current_week = weekly_by_category[category]
            # For trend calculation, we'd need historical data - simplified for now
            trends[category] = {
                'current_week': current_week,
                'trend': 'stable',  # Could be 'increasing', 'decreasing', 'stable'
                'change_percent': 0
            }
        
        # Risk assessment
        total_critical_events = sum(1 for event in all_events 
                                  if event.get('confidence', 0) >= 0.85 or 
                                     event.get('metadata', {}).get('severity') in ['high', 'critical'])
        
        risk_level = 'low'
        if total_critical_events > 20:
            risk_level = 'critical'
        elif total_critical_events > 10:
            risk_level = 'high'
        elif total_critical_events > 5:
            risk_level = 'medium'
        
        # Most recent events by category
        recent_events_by_category = {}
        for category in event_type_mapping.keys():
            category_events = []
            for event in all_events:
                event_type = event.get('event_type', event.get('type', 'unknown')).lower()
                if any(keyword in event_type for keyword in event_type_mapping[category]):
                    category_events.append(event)
            
            # Sort by timestamp and get most recent
            category_events.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            recent_events_by_category[category] = category_events[:3]  # Top 3 recent
        
        # Comprehensive analytics response
        analytics = {
            'overview': {
                'total_events': len(all_events),
                'harmful_events': len(harmful_events),
                'database_events': len(events),
                'risk_level': risk_level,
                'last_updated': datetime.now().isoformat()
            },
            'event_categories': {
                'normalized_counts': normalized_counts,
                'raw_counts': raw_event_counts,
                'severity_distribution': severity_counts
            },
            'time_analysis': {
                'daily_events_last_week': daily_events,
                'hourly_distribution': hourly_distribution,
                'weekly_by_category': weekly_by_category,
                'monthly_by_category': monthly_by_category
            },
            'trends': trends,
            'recent_events_by_category': recent_events_by_category,
            'critical_insights': {
                'most_frequent_event': max(normalized_counts.items(), key=lambda x: x[1])[0] if normalized_counts else 'none',
                'peak_hour': max(hourly_distribution.items(), key=lambda x: x[1])[0] if any(hourly_distribution.values()) else '0',
                'total_critical_events': total_critical_events,
                'events_this_week': sum(weekly_by_category.values()),
                'events_this_month': sum(monthly_by_category.values())
            }
        }
        
        return JSONResponse(content=analytics)
        
    except Exception as e:
        logger.error(f"Error getting comprehensive analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/detailed/{category}")
async def get_detailed_category_analytics(category: str):
    """Get detailed analytics for a specific event category"""
    try:
        # Validate category
        valid_categories = ['fall', 'crash', 'fire', 'fight', 'weapon', 'intrusion', 'smoke', 'medical', 'theft', 'explosion']
        if category not in valid_categories:
            raise HTTPException(status_code=400, detail=f"Invalid category. Must be one of: {valid_categories}")
        
        # Get events for this category
        all_events = db_manager.get_events(limit=5000)
        
        # Also check harmful events log
        harmful_events = []
        harmful_log_path = Path(config.database.local_media_path) / "harmful_events_log" / "harmful_events_log.json"
        if harmful_log_path.exists():
            try:
                with open(harmful_log_path, 'r') as f:
                    harmful_data = json.load(f)
                    harmful_events = harmful_data.get('events', [])
            except Exception as e:
                logger.warning(f"Could not read harmful events log: {e}")
        
        all_events.extend(harmful_events)
        
        # Filter events for this category
        category_keywords = {
            'fall': ['fall', 'falldown', 'falling'],
            'crash': ['crash', 'vehicle_crash', 'accident', 'collision'],
            'fire': ['fire', 'flame', 'burning'],
            'fight': ['fight', 'violence', 'fighting', 'assault'],
            'weapon': ['weapon_detected', 'gun', 'knife', 'rifle', 'pistol'],
            'intrusion': ['intrusion', 'trespassing', 'unauthorized_entry'],
            'smoke': ['smoke', 'smoking'],
            'medical': ['medical_emergency', 'health_emergency'],
            'theft': ['theft', 'robbery', 'burglary', 'stealing'],
            'explosion': ['explosion', 'blast']
        }
        
        keywords = category_keywords.get(category, [])
        category_events = []
        
        for event in all_events:
            event_type = event.get('event_type', event.get('type', 'unknown')).lower()
            if any(keyword in event_type for keyword in keywords):
                category_events.append(event)
        
        # Detailed analysis for this category
        confidence_distribution = {'very_high': 0, 'high': 0, 'medium': 0, 'low': 0}
        location_analysis = {}
        time_patterns = {}
        
        for event in category_events:
            # Confidence distribution
            confidence = event.get('confidence', 0)
            if confidence >= 0.9:
                confidence_distribution['very_high'] += 1
            elif confidence >= 0.8:
                confidence_distribution['high'] += 1
            elif confidence >= 0.7:
                confidence_distribution['medium'] += 1
            else:
                confidence_distribution['low'] += 1
            
            # Time pattern analysis
            timestamp = event.get('timestamp')
            if timestamp:
                try:
                    if isinstance(timestamp, str):
                        event_date = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    else:
                        event_date = timestamp
                    
                    hour = event_date.hour
                    time_period = 'morning' if 6 <= hour < 12 else 'afternoon' if 12 <= hour < 18 else 'evening' if 18 <= hour < 22 else 'night'
                    time_patterns[time_period] = time_patterns.get(time_period, 0) + 1
                except:
                    pass
        
        detailed_analytics = {
            'category': category,
            'total_events': len(category_events),
            'confidence_distribution': confidence_distribution,
            'time_patterns': time_patterns,
            'recent_events': sorted(category_events, key=lambda x: x.get('timestamp', ''), reverse=True)[:10],
            'average_confidence': sum(event.get('confidence', 0) for event in category_events) / len(category_events) if category_events else 0,
            'keywords_matched': keywords
        }
        
        return JSONResponse(content=detailed_analytics)
        
    except Exception as e:
        logger.error(f"Error getting detailed category analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, config.system.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.system.log_file),
            logging.StreamHandler()
        ]
    )
    
    # Create required directories
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    os.makedirs(config.database.local_media_path, exist_ok=True)
    
    # Run the application
    uvicorn.run(
        app,
        host=config.web.host,
        port=config.web.port,
        log_level=config.system.log_level.lower()
    )
