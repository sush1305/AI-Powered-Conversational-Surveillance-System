# AI-Powered Conversational Surveillance System

A comprehensive real-time surveillance system that combines Computer Vision, Deep Learning, and Conversational AI for intelligent event detection, natural language querying, and automated alerting.

## 🚀 Features

### Core Capabilities
- **Real-time Video Processing**: OpenCV-based continuous monitoring with 30 FPS support
- **AI Event Detection**: 
  - Person fall detection using MediaPipe pose estimation
  - Violence/weapon detection using YOLOv8n
  - Vehicle crash detection with motion analysis
  - Suspicious activity monitoring
- **Natural Language Interface**: Query surveillance history using conversational AI
- **Voice Commands**: Voice-activated queries and system control
- **Multi-channel Alerts**: SMS, Email, and Push notifications
- **Semantic Search**: FAISS-powered vector search for intelligent event matching
- **Analytics Dashboard**: Real-time statistics and data visualization

### Technical Features
- **CPU Optimized**: Runs efficiently on low-cost hardware including Raspberry Pi
- **Offline Capable**: Core functionality works without internet connection
- **Scalable Storage**: Local file system or AWS S3 integration
- **Docker Support**: Easy deployment with containerization
- **Web Dashboard**: Modern, responsive interface with real-time updates
- **Database Management**: SQLite for metadata, configurable media storage

## 📋 Requirements

### Hardware
- **Minimum**: 4GB RAM, dual-core CPU, webcam
- **Recommended**: 8GB RAM, quad-core CPU, dedicated camera
- **Supported Platforms**: Windows, Linux, macOS, Raspberry Pi

### Software Dependencies
- Python 3.8+
- OpenCV 4.8+
- PyTorch (CPU version)
- FastAPI for web interface
- SQLite for database

## 🛠️ Installation

### Quick Start (Local)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ai-surveillance-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Run the system**
   ```bash
   python main.py
   ```

5. **Access the dashboard**
   Open http://localhost:8000 in your browser

### Docker Deployment

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up -d
   ```

2. **Access the system**
   - Dashboard: http://localhost:8000
   - Logs: `docker-compose logs -f`

### Raspberry Pi Setup

1. **Install system dependencies**
   ```bash
   sudo apt update
   sudo apt install python3-pip python3-venv libgl1-mesa-glx
   ```

2. **Follow standard installation steps**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python main.py
   ```

## ⚙️ Configuration

### Environment Variables (.env)
```env
# Notifications
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_token
TWILIO_PHONE_NUMBER=+1234567890
EMAIL_ADDRESS=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
PUSHBULLET_API_KEY=your_pushbullet_key

# Storage (Optional)
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_S3_BUCKET=your_bucket_name

# Security
SECRET_KEY=your_secret_key
```

### System Configuration (config.py)
- **Video Settings**: Camera source, resolution, FPS
- **Detection Settings**: Confidence thresholds, enabled detectors
- **AI Settings**: Model paths, embedding models
- **Notification Settings**: Alert channels and recipients

## 🎯 Usage

### Web Dashboard
1. **Live Video Feed**: Monitor real-time camera feed with event overlays
2. **Natural Language Queries**: Ask questions like:
   - "What happened at 2 PM today?"
   - "Show me all falls from yesterday"
   - "Any suspicious activity this morning?"
3. **Voice Commands**: Use voice interface with wake word "surveillance"
4. **Event Management**: View, search, and export event history
5. **System Analytics**: Real-time statistics and visualizations

### Voice Interface
- **Wake Word**: "Surveillance"
- **Example Commands**:
  - "Surveillance, what happened today?"
  - "Surveillance, show recent events"
  - "Surveillance, system status"

### API Endpoints
- `GET /events` - Retrieve events with filtering
- `POST /query` - Process natural language queries
- `GET /video_feed` - Live video stream
- `GET /statistics` - System analytics
- `POST /test_notifications` - Test alert channels

## 🔧 Event Detection

### Fall Detection
- Uses MediaPipe pose estimation
- Analyzes body orientation and position
- Configurable confidence thresholds
- Real-time processing at 30 FPS

### Violence/Weapon Detection
- YOLOv8n object detection
- Identifies weapons (guns, knives)
- Motion analysis for suspicious behavior
- Contextual threat assessment

### Vehicle Crash Detection
- Multi-object tracking
- Sudden movement analysis
- Collision pattern recognition
- Configurable sensitivity

## 📊 Analytics & Reporting

### Real-time Dashboard
- Live event feed with confidence scores
- System status monitoring
- Performance metrics
- Interactive charts and graphs

### Data Export
- CSV export of event history
- Configurable date ranges
- Metadata and media file references
- Automated backup options

### Statistics
- Event counts by type and time
- Detection accuracy metrics
- System uptime and performance
- Alert delivery status

## 🔔 Notification System

### SMS Alerts (Twilio)
- Instant critical event notifications
- Configurable recipient lists
- Rate limiting to prevent spam
- Delivery confirmation

### Email Notifications
- Detailed event reports with images
- HTML formatted messages
- Attachment support for media files
- SMTP configuration for any provider

### Push Notifications (Pushbullet)
- Real-time mobile alerts
- Rich notifications with images
- Cross-platform support
- Silent hours configuration

## 🤖 AI & Machine Learning

### Natural Language Processing
- LangChain integration for query parsing
- Llama.cpp for local LLM inference
- Semantic search with sentence transformers
- Context-aware response generation

### Computer Vision Models
- **YOLOv8n**: Lightweight object detection
- **MediaPipe**: Real-time pose estimation
- **OpenVINO**: Optimized inference (optional)
- **Custom Models**: Extensible architecture

### Semantic Search
- FAISS vector database
- Sentence transformer embeddings
- Similarity-based event matching
- Incremental index updates

## 🔒 Security & Privacy

### Data Protection
- Local-first architecture
- Encrypted database storage
- Configurable data retention
- GDPR compliance features

### Access Control
- Web dashboard authentication
- API key management
- Role-based permissions
- Audit logging

### Network Security
- HTTPS support with SSL certificates
- Firewall configuration guidelines
- VPN integration support
- Secure API endpoints

## 🚀 Deployment Options

### Local Development
```bash
python main.py
```

### Production Docker
```bash
docker-compose -f docker-compose.prod.yml up -d
```

### Raspberry Pi Service
```bash
sudo systemctl enable surveillance-system
sudo systemctl start surveillance-system
```

### Cloud Deployment
- AWS EC2 with GPU support
- Google Cloud Platform
- Azure Container Instances
- DigitalOcean Droplets

## 🔧 Troubleshooting

### Common Issues

**Camera not detected**
```bash
# Check camera permissions
ls /dev/video*
# Test camera access
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

**Model loading errors**
```bash
# Download required models
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

**Memory issues**
- Reduce video resolution in config.py
- Lower detection confidence thresholds
- Enable frame skipping for processing

**Network connectivity**
- Check firewall settings
- Verify port 8000 is available
- Test WebSocket connections

### Performance Optimization

**CPU Usage**
- Use OpenVINO for Intel hardware
- Reduce video FPS
- Enable frame skipping
- Use smaller AI models

**Memory Usage**
- Limit event buffer size
- Configure automatic cleanup
- Use efficient video codecs
- Optimize database queries

**Storage Management**
- Enable automatic file cleanup
- Use video compression
- Configure S3 archiving
- Monitor disk usage

## 📈 Monitoring & Maintenance

### Health Checks
- System status endpoint: `/system_status`
- Database connectivity tests
- Camera feed validation
- AI model performance metrics

### Logging
- Structured JSON logging
- Configurable log levels
- Automatic log rotation
- Error tracking and alerts

### Updates
- Automated dependency updates
- Model version management
- Configuration migration
- Backup and restore procedures

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Install development dependencies
4. Run tests: `pytest`
5. Submit a pull request

### Code Standards
- PEP 8 compliance
- Type hints required
- Comprehensive docstrings
- Unit test coverage >80%

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenCV** - Computer vision library
- **Ultralytics** - YOLOv8 implementation
- **MediaPipe** - Real-time pose detection
- **LangChain** - LLM application framework
- **FastAPI** - Modern web framework
- **Bootstrap** - UI components

## 📞 Support

- **Documentation**: [Wiki](wiki)
- **Issues**: [GitHub Issues](issues)
- **Discussions**: [GitHub Discussions](discussions)
- **Email**: support@surveillance-ai.com

---

**Built with ❤️ for intelligent surveillance and security**
