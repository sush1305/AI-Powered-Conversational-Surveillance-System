import os
from dataclasses import dataclass
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()

@dataclass
class VideoConfig:
    """Video processing configuration"""
    source: str = "0"  # Use camera 0 (confirmed working by diagnostic test)
    fps: int = 30
    resolution: tuple = (640, 480)
    record_buffer_seconds: int = 10
    event_clip_duration: int = 30

@dataclass
class DetectionConfig:
    """Event detection configuration"""
    yolo_model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.5
    fall_detection_enabled: bool = True
    violence_detection_enabled: bool = True
    vehicle_crash_detection_enabled: bool = True
    
@dataclass
class DatabaseConfig:
    """Database configuration"""
    sqlite_path: str = "surveillance.db"
    media_storage_type: str = "local"  # "local" or "s3"
    local_media_path: str = "media"
    aws_s3_bucket: Optional[str] = None
    aws_access_key: Optional[str] = None
    aws_secret_key: Optional[str] = None

@dataclass
class AIConfig:
    """AI and NLP configuration"""
    llama_model_path: str = "models/llama-2-7b-chat.gguf"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_db_type: str = "faiss"  # "faiss" or "chromadb"
    vector_db_path: str = "vector_db"

@dataclass
class NotificationConfig:
    """Notification configuration"""
    # Twilio SMS
    twilio_account_sid: Optional[str] = os.getenv("TWILIO_ACCOUNT_SID")
    twilio_auth_token: Optional[str] = os.getenv("TWILIO_AUTH_TOKEN")
    twilio_phone_number: Optional[str] = os.getenv("TWILIO_PHONE_NUMBER")
    alert_phone_numbers: List[str] = None
    
    # Email
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    email_address: Optional[str] = os.getenv("EMAIL_ADDRESS")
    email_password: Optional[str] = os.getenv("EMAIL_PASSWORD")
    alert_emails: List[str] = None
    
    # Pushbullet
    pushbullet_api_key: Optional[str] = os.getenv("PUSHBULLET_API_KEY")
    
    def __post_init__(self):
        if self.alert_phone_numbers is None:
            self.alert_phone_numbers = []
        if self.alert_emails is None:
            self.alert_emails = []

@dataclass
class WebConfig:
    """Web dashboard configuration"""
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False
    secret_key: str = os.getenv("SECRET_KEY", "your-secret-key-here")

@dataclass
class SystemConfig:
    """System-wide configuration"""
    log_level: str = "INFO"
    log_file: str = "surveillance.log"
    auto_restart: bool = True
    max_events_in_memory: int = 1000
    cleanup_old_events_days: int = 30

class Config:
    """Main configuration class"""
    def __init__(self):
        self.video = VideoConfig()
        self.detection = DetectionConfig()
        self.database = DatabaseConfig()
        self.ai = AIConfig()
        self.notification = NotificationConfig()
        self.web = WebConfig()
        self.system = SystemConfig()
        
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Check if required model files exist
        if not os.path.exists(self.ai.llama_model_path):
            issues.append(f"Llama model not found at {self.ai.llama_model_path}")
            
        # Check notification settings
        if not any([
            self.notification.twilio_account_sid,
            self.notification.email_address,
            self.notification.pushbullet_api_key
        ]):
            issues.append("No notification methods configured")
            
        return issues

# Global config instance
config = Config()
