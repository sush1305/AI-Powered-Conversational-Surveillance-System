#!/usr/bin/env python3
"""
AI Surveillance System Setup Script
Automated installation and configuration
"""

import os
import sys
import subprocess
import platform
import urllib.request
from pathlib import Path

def print_banner():
    """Print setup banner"""
    print("=" * 70)
    print("AI-Powered Conversational Surveillance System")
    print("Automated Setup Script")
    print("=" * 70)
    print()

def check_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("[ERROR] Python 3.8+ is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"[OK] Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_system_dependencies():
    """Install system-level dependencies"""
    print("\n📦 Installing system dependencies...")
    
    system = platform.system().lower()
    
    if system == "linux":
        # Ubuntu/Debian
        commands = [
            "sudo apt update",
            "sudo apt install -y python3-pip python3-venv",
            "sudo apt install -y libgl1-mesa-glx libglib2.0-0",
            "sudo apt install -y libsm6 libxext6 libxrender-dev",
            "sudo apt install -y libgstreamer1.0-0 libgstreamer-plugins-base1.0-0",
            "sudo apt install -y libavcodec-dev libavformat-dev libswscale-dev",
            "sudo apt install -y portaudio19-dev python3-pyaudio"
        ]
        
        for cmd in commands:
            print(f"Running: {cmd}")
            try:
                subprocess.run(cmd.split(), check=True)
            except subprocess.CalledProcessError:
                print(f"[WARNING]  Warning: Failed to run {cmd}")
                
    elif system == "darwin":  # macOS
        print("🍎 macOS detected - please install dependencies manually:")
        print("  brew install portaudio")
        print("  brew install opencv")
        
    elif system == "windows":
        print("🪟 Windows detected - dependencies will be installed via pip")
        
    print("[OK] System dependencies installation completed")

def create_virtual_environment():
    """Create Python virtual environment"""
    print("\n🐍 Creating virtual environment...")
    
    venv_path = Path("venv")
    if venv_path.exists():
        print("Virtual environment already exists")
        return True
        
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("[OK] Virtual environment created")
        return True
    except subprocess.CalledProcessError:
        print("[ERROR] Failed to create virtual environment")
        return False

def install_python_dependencies():
    """Install Python package dependencies"""
    print("\n📚 Installing Python dependencies...")
    
    # Determine pip executable
    system = platform.system().lower()
    if system == "windows":
        pip_exe = "venv\\Scripts\\pip.exe"
        python_exe = "venv\\Scripts\\python.exe"
    else:
        pip_exe = "venv/bin/pip"
        python_exe = "venv/bin/python"
    
    # Upgrade pip first
    try:
        subprocess.run([python_exe, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        print("[OK] Pip upgraded")
    except subprocess.CalledProcessError:
        print("[WARNING]  Warning: Failed to upgrade pip")
    
    # Install requirements
    try:
        subprocess.run([pip_exe, "install", "-r", "requirements.txt"], check=True)
        print("[OK] Python dependencies installed")
        return True
    except subprocess.CalledProcessError:
        print("[ERROR] Failed to install Python dependencies")
        return False

def download_ai_models():
    """Download required AI models"""
    print("\n[AI] Downloading AI models...")
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Download YOLOv8n model
    try:
        print("Downloading YOLOv8n model...")
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        print("[OK] YOLOv8n model downloaded")
    except Exception as e:
        print(f"[WARNING]  Warning: Failed to download YOLOv8n: {e}")
    
    # Download sentence transformer model
    try:
        print("Downloading sentence transformer model...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("[OK] Sentence transformer model downloaded")
    except Exception as e:
        print(f"[WARNING]  Warning: Failed to download sentence transformer: {e}")
    
    # Download Llama model
    llama_model_path = models_dir / "llama-2-7b-chat.gguf"
    if not llama_model_path.exists():
        try:
            print("Downloading Llama-2-7B-Chat model (this may take a while)...")
            import urllib.request
            import shutil
            
            url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"
            
            # Download with progress
            def show_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(100, downloaded * 100 / total_size)
                    print(f"\rDownloading... {percent:.1f}% ({downloaded // (1024*1024)} MB / {total_size // (1024*1024)} MB)", end="")
            
            urllib.request.urlretrieve(url, llama_model_path, reporthook=show_progress)
            print("\n[OK] Llama model downloaded successfully")
            
        except Exception as e:
            print(f"\n[WARNING]  Warning: Failed to download Llama model: {e}")
            print("Please manually download from:")
            print("https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf")
            print(f"And save as: {llama_model_path}")
    else:
        print("[OK] Llama model already exists")
    
    print("[OK] AI models setup completed")

def create_directories():
    """Create required directories"""
    print("\n[DIR] Creating directories...")
    
    directories = [
        "media",
        "logs", 
        "models",
        "vector_db",
        "templates",
        "static"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"[OK] Created {directory}/")

def setup_configuration():
    """Setup configuration files"""
    print("\n[CONFIG]  Setting up configuration...")
    
    # Copy .env.example to .env if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        env_example = Path(".env.example")
        if env_example.exists():
            env_file.write_text(env_example.read_text())
            print("[OK] Created .env file from template")
        else:
            print("[WARNING]  Warning: .env.example not found")
    else:
        print("[OK] .env file already exists")

def test_installation():
    """Test the installation"""
    print("\n🧪 Testing installation...")
    
    # Determine python executable
    system = platform.system().lower()
    if system == "windows":
        python_exe = "venv\\Scripts\\python.exe"
    else:
        python_exe = "venv/bin/python"
    
    # Test imports
    test_script = """
import sys
try:
    import cv2
    print("[OK] OpenCV imported successfully")
except ImportError as e:
    print(f"[ERROR] OpenCV import failed: {e}")
    sys.exit(1)

try:
    import torch
    print("[OK] PyTorch imported successfully")
except ImportError as e:
    print(f"[ERROR] PyTorch import failed: {e}")
    sys.exit(1)

try:
    from ultralytics import YOLO
    print("[OK] Ultralytics imported successfully")
except ImportError as e:
    print(f"[ERROR] Ultralytics import failed: {e}")
    sys.exit(1)

try:
    import fastapi
    print("[OK] FastAPI imported successfully")
except ImportError as e:
    print(f"[ERROR] FastAPI import failed: {e}")
    sys.exit(1)

print("[OK] All core dependencies are working!")
"""
    
    try:
        result = subprocess.run([python_exe, "-c", test_script], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("[ERROR] Installation test failed:")
        print(e.stdout)
        print(e.stderr)
        return False

def print_next_steps():
    """Print next steps for the user"""
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next Steps:")
    print("1. Edit .env file with your configuration:")
    print("   - Add Twilio credentials for SMS alerts")
    print("   - Add email settings for notifications")
    print("   - Add Pushbullet API key for push notifications")
    print()
    print("2. Start the surveillance system:")
    
    system = platform.system().lower()
    if system == "windows":
        print("   venv\\Scripts\\python.exe main.py")
    else:
        print("   source venv/bin/activate")
        print("   python main.py")
    
    print()
    print("3. Access the web dashboard:")
    print("   http://localhost:8000")
    print()
    print("4. For Docker deployment:")
    print("   docker-compose up -d")
    print()
    print("📖 Read README.md for detailed documentation")
    print("🐛 Report issues at: https://github.com/your-repo/issues")

def main():
    """Main setup function"""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install system dependencies
    install_system_dependencies()
    
    # Create virtual environment
    if not create_virtual_environment():
        sys.exit(1)
    
    # Install Python dependencies
    if not install_python_dependencies():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Setup configuration
    setup_configuration()
    
    # Download AI models
    download_ai_models()
    
    # Test installation
    if not test_installation():
        print("\n[WARNING]  Installation test failed, but you can try running the system anyway")
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()
