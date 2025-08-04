import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from typing import List, Dict, Optional
from datetime import datetime
import requests
import json
from twilio.rest import Client
from pushbullet import Pushbullet
from config import config

logger = logging.getLogger(__name__)

class NotificationManager:
    """Manages all notification channels for surveillance alerts"""
    
    def __init__(self):
        self.sms_client = None
        self.email_config = None
        self.pushbullet_client = None
        
        self._initialize_clients()
        
    def _initialize_clients(self):
        """Initialize notification service clients"""
        # Initialize Twilio SMS
        if (config.notification.twilio_account_sid and 
            config.notification.twilio_auth_token):
            try:
                self.sms_client = Client(
                    config.notification.twilio_account_sid,
                    config.notification.twilio_auth_token
                )
                logger.info("Twilio SMS client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Twilio client: {e}")
                
        # Initialize Email
        if config.notification.email_address and config.notification.email_password:
            self.email_config = {
                'server': config.notification.smtp_server,
                'port': config.notification.smtp_port,
                'email': config.notification.email_address,
                'password': config.notification.email_password
            }
            logger.info("Email client configured")
            
        # Initialize Pushbullet
        if config.notification.pushbullet_api_key:
            try:
                self.pushbullet_client = Pushbullet(config.notification.pushbullet_api_key)
                logger.info("Pushbullet client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Pushbullet client: {e}")
                
    def send_alert(self, event: Dict, media_path: Optional[str] = None):
        """Send alert through all configured channels"""
        alert_message = self._create_alert_message(event)
        
        # Send SMS alerts
        if self.sms_client and config.notification.alert_phone_numbers:
            self._send_sms_alerts(alert_message, config.notification.alert_phone_numbers)
            
        # Send email alerts
        if self.email_config and config.notification.alert_emails:
            self._send_email_alerts(alert_message, config.notification.alert_emails, media_path)
            
        # Send push notifications
        if self.pushbullet_client:
            self._send_push_notification(alert_message)
            
    def _create_alert_message(self, event: Dict) -> str:
        """Create formatted alert message"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        message = f"""
🚨 SURVEILLANCE ALERT 🚨

Event Type: {event.get('type', 'Unknown').replace('_', ' ').title()}
Time: {timestamp}
Description: {event.get('description', 'No description available')}
Confidence: {event.get('confidence', 0):.2f}

Location: Camera Feed
System: AI Surveillance System
        """.strip()
        
        return message
        
    def _send_sms_alerts(self, message: str, phone_numbers: List[str]):
        """Send SMS alerts via Twilio"""
        if not self.sms_client:
            return
            
        for phone_number in phone_numbers:
            try:
                self.sms_client.messages.create(
                    body=message,
                    from_=config.notification.twilio_phone_number,
                    to=phone_number
                )
                logger.info(f"SMS alert sent to {phone_number}")
                
            except Exception as e:
                logger.error(f"Failed to send SMS to {phone_number}: {e}")
                
    def _send_email_alerts(self, message: str, email_addresses: List[str], 
                          media_path: Optional[str] = None):
        """Send email alerts with optional media attachment"""
        if not self.email_config:
            return
            
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_config['email']
            msg['Subject'] = "🚨 Surveillance Alert - Event Detected"
            
            # Add text content
            msg.attach(MIMEText(message, 'plain'))
            
            # Add media attachment if available
            if media_path and media_path.startswith('/') or media_path.startswith('C:'):
                try:
                    with open(media_path, 'rb') as f:
                        img_data = f.read()
                        image = MIMEImage(img_data)
                        image.add_header('Content-Disposition', 
                                       f'attachment; filename="event_image.jpg"')
                        msg.attach(image)
                except Exception as e:
                    logger.error(f"Failed to attach media: {e}")
                    
            # Send to each recipient
            for email_address in email_addresses:
                try:
                    msg['To'] = email_address
                    
                    # Connect to server and send
                    server = smtplib.SMTP(self.email_config['server'], 
                                        self.email_config['port'])
                    server.starttls()
                    server.login(self.email_config['email'], 
                               self.email_config['password'])
                    
                    text = msg.as_string()
                    server.sendmail(self.email_config['email'], email_address, text)
                    server.quit()
                    
                    logger.info(f"Email alert sent to {email_address}")
                    
                except Exception as e:
                    logger.error(f"Failed to send email to {email_address}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to create email message: {e}")
            
    def _send_push_notification(self, message: str):
        """Send push notification via Pushbullet"""
        if not self.pushbullet_client:
            return
            
        try:
            self.pushbullet_client.push_note(
                "🚨 Surveillance Alert", 
                message
            )
            logger.info("Push notification sent via Pushbullet")
            
        except Exception as e:
            logger.error(f"Failed to send push notification: {e}")
            
    def test_notifications(self) -> Dict[str, bool]:
        """Test all notification channels"""
        results = {
            'sms': False,
            'email': False,
            'push': False
        }
        
        test_message = "Test message from AI Surveillance System"
        
        # Test SMS
        if self.sms_client and config.notification.alert_phone_numbers:
            try:
                phone_number = config.notification.alert_phone_numbers[0]
                self.sms_client.messages.create(
                    body=test_message,
                    from_=config.notification.twilio_phone_number,
                    to=phone_number
                )
                results['sms'] = True
                logger.info("SMS test successful")
            except Exception as e:
                logger.error(f"SMS test failed: {e}")
                
        # Test Email
        if self.email_config and config.notification.alert_emails:
            try:
                email_address = config.notification.alert_emails[0]
                
                msg = MIMEText(test_message)
                msg['Subject'] = "Test - AI Surveillance System"
                msg['From'] = self.email_config['email']
                msg['To'] = email_address
                
                server = smtplib.SMTP(self.email_config['server'], 
                                    self.email_config['port'])
                server.starttls()
                server.login(self.email_config['email'], 
                           self.email_config['password'])
                server.sendmail(self.email_config['email'], email_address, 
                              msg.as_string())
                server.quit()
                
                results['email'] = True
                logger.info("Email test successful")
                
            except Exception as e:
                logger.error(f"Email test failed: {e}")
                
        # Test Push notification
        if self.pushbullet_client:
            try:
                self.pushbullet_client.push_note("Test - AI Surveillance", test_message)
                results['push'] = True
                logger.info("Push notification test successful")
                
            except Exception as e:
                logger.error(f"Push notification test failed: {e}")
                
        return results
        
    def send_system_notification(self, title: str, message: str, level: str = "INFO"):
        """Send system status notifications"""
        formatted_message = f"""
System Notification - {level}

{title}

{message}

Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
System: AI Surveillance System
        """.strip()
        
        # Only send system notifications via push and email (not SMS)
        if self.email_config and config.notification.alert_emails:
            try:
                msg = MIMEText(formatted_message)
                msg['Subject'] = f"System {level} - {title}"
                msg['From'] = self.email_config['email']
                
                for email_address in config.notification.alert_emails:
                    msg['To'] = email_address
                    
                    server = smtplib.SMTP(self.email_config['server'], 
                                        self.email_config['port'])
                    server.starttls()
                    server.login(self.email_config['email'], 
                               self.email_config['password'])
                    server.sendmail(self.email_config['email'], email_address, 
                                  msg.as_string())
                    server.quit()
                    
            except Exception as e:
                logger.error(f"Failed to send system notification email: {e}")
                
        if self.pushbullet_client:
            try:
                self.pushbullet_client.push_note(f"System {level}", formatted_message)
            except Exception as e:
                logger.error(f"Failed to send system push notification: {e}")

class AlertThrottler:
    """Prevents spam by throttling similar alerts"""
    
    def __init__(self, cooldown_seconds: int = 300):  # 5 minute cooldown
        self.cooldown_seconds = cooldown_seconds
        self.recent_alerts = {}
        
    def should_send_alert(self, event_type: str, location: str = "default") -> bool:
        """Check if alert should be sent based on throttling rules"""
        key = f"{event_type}_{location}"
        current_time = datetime.now().timestamp()
        
        if key in self.recent_alerts:
            time_since_last = current_time - self.recent_alerts[key]
            if time_since_last < self.cooldown_seconds:
                return False
                
        self.recent_alerts[key] = current_time
        return True
        
    def reset_throttle(self, event_type: str = None, location: str = None):
        """Reset throttling for specific event type or all events"""
        if event_type and location:
            key = f"{event_type}_{location}"
            if key in self.recent_alerts:
                del self.recent_alerts[key]
        elif event_type:
            # Reset all alerts for this event type
            keys_to_remove = [k for k in self.recent_alerts.keys() 
                            if k.startswith(f"{event_type}_")]
            for key in keys_to_remove:
                del self.recent_alerts[key]
        else:
            # Reset all alerts
            self.recent_alerts.clear()

class NotificationScheduler:
    """Schedules and manages notification timing"""
    
    def __init__(self):
        self.quiet_hours = (22, 6)  # 10 PM to 6 AM
        self.emergency_events = ['fall', 'weapon_detected', 'vehicle_crash']
        
    def should_send_immediate_alert(self, event: Dict) -> bool:
        """Determine if alert should be sent immediately"""
        event_type = event.get('type', '')
        
        # Always send emergency alerts
        if event_type in self.emergency_events:
            return True
            
        # Check quiet hours for non-emergency events
        current_hour = datetime.now().hour
        if self.quiet_hours[0] <= current_hour or current_hour <= self.quiet_hours[1]:
            return False
            
        return True
        
    def get_next_notification_time(self, event: Dict) -> datetime:
        """Get the next appropriate time to send notification"""
        if self.should_send_immediate_alert(event):
            return datetime.now()
            
        # Schedule for after quiet hours
        next_morning = datetime.now().replace(hour=self.quiet_hours[1], 
                                            minute=0, second=0, microsecond=0)
        if datetime.now().hour >= self.quiet_hours[1]:
            # Add one day if we're past morning quiet hour end
            next_morning = next_morning.replace(day=next_morning.day + 1)
            
        return next_morning

# Global instances
notification_manager = NotificationManager()
alert_throttler = AlertThrottler()
notification_scheduler = NotificationScheduler()
