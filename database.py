import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import boto3
from botocore.exceptions import ClientError
import logging
from config import config

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages SQLite database operations for surveillance events"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.database.sqlite_path
        self.init_database()
        
    def init_database(self):
        """Initialize the database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    event_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    media_path TEXT,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Event embeddings for semantic search
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS event_embeddings (
                    event_id INTEGER PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    FOREIGN KEY (event_id) REFERENCES events (id)
                )
            ''')
            
            # System logs
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    component TEXT
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)')
            
            conn.commit()
            
    def add_event(self, event_type: str, description: str, confidence_score: float, 
                  media_path: str = None, metadata: Dict = None) -> int:
        """Add a new event to the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO events (timestamp, event_type, description, confidence_score, media_path, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                event_type,
                description,
                confidence_score,
                media_path,
                json.dumps(metadata) if metadata else None
            ))
            
            event_id = cursor.lastrowid
            conn.commit()
            logger.info(f"Added event {event_id}: {event_type} - {description}")
            return event_id
            
    def get_events(self, start_time: datetime = None, end_time: datetime = None, 
                   event_type: str = None, limit: int = 100) -> List[Dict]:
        """Retrieve events with optional filtering"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = "SELECT * FROM events WHERE 1=1"
            params = []
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
                
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)
                
            if event_type:
                query += " AND event_type = ?"
                params.append(event_type)
                
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            events = []
            
            for row in cursor.fetchall():
                event = dict(row)
                if event['metadata']:
                    event['metadata'] = json.loads(event['metadata'])
                events.append(event)
                
            return events
            
    def search_events_by_text(self, search_text: str, limit: int = 50) -> List[Dict]:
        """Search events by description text"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM events 
                WHERE description LIKE ? OR event_type LIKE ?
                ORDER BY timestamp DESC LIMIT ?
            ''', (f'%{search_text}%', f'%{search_text}%', limit))
            
            events = []
            for row in cursor.fetchall():
                event = dict(row)
                if event['metadata']:
                    event['metadata'] = json.loads(event['metadata'])
                events.append(event)
                
            return events
            
    def add_embedding(self, event_id: int, embedding: bytes):
        """Add embedding for an event"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO event_embeddings (event_id, embedding)
                VALUES (?, ?)
            ''', (event_id, embedding))
            conn.commit()
            
    def get_embedding(self, event_id: int) -> Optional[bytes]:
        """Get embedding for an event"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT embedding FROM event_embeddings WHERE event_id = ?', (event_id,))
            result = cursor.fetchone()
            return result[0] if result else None
            
    def cleanup_old_events(self, days: int = 30):
        """Remove events older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get old event IDs first
            cursor.execute('SELECT id FROM events WHERE timestamp < ?', (cutoff_date,))
            old_event_ids = [row[0] for row in cursor.fetchall()]
            
            # Delete old events and their embeddings
            cursor.execute('DELETE FROM events WHERE timestamp < ?', (cutoff_date,))
            cursor.execute('DELETE FROM event_embeddings WHERE event_id IN ({})'.format(
                ','.join('?' * len(old_event_ids))
            ), old_event_ids)
            
            conn.commit()
            logger.info(f"Cleaned up {len(old_event_ids)} old events")
            
    def log_system_event(self, level: str, message: str, component: str = None):
        """Log system events"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO system_logs (level, message, component)
                VALUES (?, ?, ?)
            ''', (level, message, component))
            conn.commit()

class MediaStorage:
    """Handles media file storage (local or S3)"""
    
    def __init__(self):
        self.storage_type = config.database.media_storage_type
        self.local_path = config.database.local_media_path
        
        if self.storage_type == "s3":
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=config.database.aws_access_key,
                aws_secret_access_key=config.database.aws_secret_key
            )
            self.bucket_name = config.database.aws_s3_bucket
        else:
            os.makedirs(self.local_path, exist_ok=True)
            
    def save_media(self, media_data: bytes, filename: str) -> str:
        """Save media file and return path/URL"""
        if self.storage_type == "local":
            file_path = os.path.join(self.local_path, filename)
            with open(file_path, 'wb') as f:
                f.write(media_data)
            return file_path
        else:
            try:
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=filename,
                    Body=media_data
                )
                return f"s3://{self.bucket_name}/{filename}"
            except ClientError as e:
                logger.error(f"Failed to upload to S3: {e}")
                raise
                
    def get_media_url(self, media_path: str) -> str:
        """Get URL for media file"""
        if self.storage_type == "local":
            return f"/media/{os.path.basename(media_path)}"
        else:
            # Generate presigned URL for S3
            try:
                key = media_path.replace(f"s3://{self.bucket_name}/", "")
                url = self.s3_client.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': self.bucket_name, 'Key': key},
                    ExpiresIn=3600
                )
                return url
            except ClientError as e:
                logger.error(f"Failed to generate presigned URL: {e}")
                return ""
                
    def delete_media(self, media_path: str):
        """Delete media file"""
        if self.storage_type == "local":
            if os.path.exists(media_path):
                os.remove(media_path)
        else:
            try:
                key = media_path.replace(f"s3://{self.bucket_name}/", "")
                self.s3_client.delete_object(Bucket=self.bucket_name, Key=key)
            except ClientError as e:
                logger.error(f"Failed to delete from S3: {e}")

# Global instances
db_manager = DatabaseManager()
media_storage = MediaStorage()
