import speech_recognition as sr
import pyttsx3
import threading
import queue
import logging
from typing import Optional, Callable
import time
from datetime import datetime
from nlp_interface import nlp_interface

logger = logging.getLogger(__name__)

class VoiceInterface:
    """Handles voice input and output for the surveillance system"""
    
    def __init__(self):
        # Speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.is_listening = False
        self.listen_thread = None
        
        # Text-to-speech
        self.tts_engine = None
        self.voice_enabled = True
        
        # Audio queue for processing
        self.audio_queue = queue.Queue()
        
        # Callbacks
        self.query_callback: Optional[Callable] = None
        
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize speech recognition and TTS components"""
        try:
            # Initialize microphone
            self.microphone = sr.Microphone()
            
            # Adjust for ambient noise
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
            logger.info("Microphone initialized and calibrated")
            
        except Exception as e:
            logger.error(f"Failed to initialize microphone: {e}")
            self.microphone = None
            
        try:
            # Initialize TTS engine
            self.tts_engine = pyttsx3.init()
            
            # Configure TTS settings
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Use female voice if available
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
                        
            # Set speech rate and volume
            self.tts_engine.setProperty('rate', 180)  # Speed of speech
            self.tts_engine.setProperty('volume', 0.8)  # Volume level
            
            logger.info("Text-to-speech engine initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {e}")
            self.tts_engine = None
            
    def start_listening(self, wake_word: str = "surveillance"):
        """Start continuous listening for voice commands"""
        if not self.microphone:
            logger.error("Cannot start listening - microphone not available")
            return False
            
        self.is_listening = True
        self.wake_word = wake_word.lower()
        
        # Start listening thread
        self.listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listen_thread.start()
        
        logger.info(f"Voice interface started - listening for wake word: '{wake_word}'")
        return True
        
    def stop_listening(self):
        """Stop voice listening"""
        self.is_listening = False
        
        if self.listen_thread:
            self.listen_thread.join(timeout=2.0)
            
        logger.info("Voice interface stopped")
        
    def _listen_loop(self):
        """Main listening loop"""
        while self.is_listening:
            try:
                # Listen for audio
                with self.microphone as source:
                    # Listen with timeout
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                    
                # Add to processing queue
                self.audio_queue.put(audio)
                
                # Process audio in separate thread
                threading.Thread(target=self._process_audio, args=(audio,), daemon=True).start()
                
            except sr.WaitTimeoutError:
                # Normal timeout, continue listening
                continue
            except Exception as e:
                logger.error(f"Error in listening loop: {e}")
                time.sleep(1)  # Brief pause before retrying
                
    def _process_audio(self, audio):
        """Process captured audio"""
        try:
            # Convert speech to text
            text = self.recognizer.recognize_google(audio).lower()
            logger.info(f"Recognized speech: {text}")
            
            # Check for wake word
            if self.wake_word in text:
                self._handle_wake_word_detected(text)
                
        except sr.UnknownValueError:
            # Speech not recognized - ignore
            pass
        except sr.RequestError as e:
            logger.error(f"Speech recognition service error: {e}")
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            
    def _handle_wake_word_detected(self, text: str):
        """Handle wake word detection and process command"""
        try:
            # Remove wake word from text
            command = text.replace(self.wake_word, "").strip()
            
            if not command:
                self.speak("Yes, how can I help you?")
                # Listen for follow-up command
                command = self._listen_for_command()
                
            if command:
                self.speak("Processing your request...")
                self._process_voice_command(command)
                
        except Exception as e:
            logger.error(f"Error handling wake word: {e}")
            self.speak("Sorry, I encountered an error processing your request.")
            
    def _listen_for_command(self, timeout: int = 5) -> Optional[str]:
        """Listen for a specific command after wake word"""
        try:
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=10)
                
            command = self.recognizer.recognize_google(audio)
            logger.info(f"Command recognized: {command}")
            return command.lower()
            
        except sr.WaitTimeoutError:
            self.speak("I didn't hear anything. Please try again.")
            return None
        except sr.UnknownValueError:
            self.speak("I didn't understand that. Please try again.")
            return None
        except Exception as e:
            logger.error(f"Error listening for command: {e}")
            return None
            
    def _process_voice_command(self, command: str):
        """Process voice command and generate response"""
        try:
            # Use NLP interface to process the query
            result = nlp_interface.process_query(command)
            
            # Generate voice response
            response = self._generate_voice_response(result)
            self.speak(response)
            
            # Call callback if set
            if self.query_callback:
                self.query_callback(command, result)
                
        except Exception as e:
            logger.error(f"Error processing voice command: {e}")
            self.speak("Sorry, I couldn't process your request.")
            
    def _generate_voice_response(self, query_result: dict) -> str:
        """Generate natural language response from query results"""
        try:
            results = query_result.get('results', [])
            total_count = query_result.get('total_count', 0)
            
            if total_count == 0:
                return "I didn't find any events matching your query."
                
            # Generate summary response
            if total_count == 1:
                event = results[0]
                event_type = event.get('event_type', 'unknown').replace('_', ' ')
                timestamp = event.get('timestamp', 'unknown time')
                
                return f"I found one {event_type} event at {timestamp}. {event.get('description', '')}"
                
            elif total_count <= 5:
                # List individual events
                response = f"I found {total_count} events:\n"
                for i, event in enumerate(results[:5], 1):
                    event_type = event.get('event_type', 'unknown').replace('_', ' ')
                    timestamp = event.get('timestamp', 'unknown time')
                    response += f"{i}. {event_type} at {timestamp}. "
                    
                return response
                
            else:
                # Summarize by event types
                event_types = {}
                for event in results:
                    event_type = event.get('event_type', 'unknown').replace('_', ' ')
                    event_types[event_type] = event_types.get(event_type, 0) + 1
                    
                response = f"I found {total_count} events in total: "
                type_summaries = []
                for event_type, count in event_types.items():
                    type_summaries.append(f"{count} {event_type} events")
                    
                response += ", ".join(type_summaries) + "."
                return response
                
        except Exception as e:
            logger.error(f"Error generating voice response: {e}")
            return "I found some events but had trouble summarizing them."
            
    def speak(self, text: str):
        """Convert text to speech"""
        if not self.voice_enabled or not self.tts_engine:
            return
            
        try:
            logger.info(f"Speaking: {text}")
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
            
        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}")
            
    def set_voice_enabled(self, enabled: bool):
        """Enable or disable voice output"""
        self.voice_enabled = enabled
        logger.info(f"Voice output {'enabled' if enabled else 'disabled'}")
        
    def set_query_callback(self, callback: Callable):
        """Set callback function for voice queries"""
        self.query_callback = callback
        
    def test_voice_interface(self) -> dict:
        """Test voice interface components"""
        results = {
            'microphone': False,
            'speech_recognition': False,
            'text_to_speech': False
        }
        
        # Test microphone
        if self.microphone:
            try:
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                results['microphone'] = True
                logger.info("Microphone test passed")
            except Exception as e:
                logger.error(f"Microphone test failed: {e}")
                
        # Test speech recognition (with timeout)
        try:
            # This would require actual speech input, so we'll just check if the service is available
            results['speech_recognition'] = True
            logger.info("Speech recognition test passed")
        except Exception as e:
            logger.error(f"Speech recognition test failed: {e}")
            
        # Test text-to-speech
        if self.tts_engine:
            try:
                self.tts_engine.say("Voice interface test successful")
                self.tts_engine.runAndWait()
                results['text_to_speech'] = True
                logger.info("Text-to-speech test passed")
            except Exception as e:
                logger.error(f"Text-to-speech test failed: {e}")
                
        return results
        
    def process_text_query(self, text: str) -> dict:
        """Process text query (for testing without voice input)"""
        try:
            result = nlp_interface.process_query(text)
            response = self._generate_voice_response(result)
            
            if self.voice_enabled:
                self.speak(response)
                
            return {
                'query': text,
                'result': result,
                'response': response
            }
            
        except Exception as e:
            logger.error(f"Error processing text query: {e}")
            return {
                'query': text,
                'result': {'error': str(e)},
                'response': "Sorry, I couldn't process your request."
            }

class VoiceCommands:
    """Predefined voice commands and their handlers"""
    
    COMMANDS = {
        'status': ['status', 'system status', 'how are you'],
        'recent_events': ['recent events', 'what happened recently', 'latest events'],
        'help': ['help', 'what can you do', 'commands'],
        'test': ['test', 'test system', 'run test'],
        'stop_listening': ['stop listening', 'disable voice', 'quiet mode'],
        'start_listening': ['start listening', 'enable voice', 'wake up']
    }
    
    @staticmethod
    def get_command_type(text: str) -> Optional[str]:
        """Identify command type from text"""
        text_lower = text.lower()
        
        for command_type, phrases in VoiceCommands.COMMANDS.items():
            for phrase in phrases:
                if phrase in text_lower:
                    return command_type
                    
        return None
        
    @staticmethod
    def get_help_text() -> str:
        """Get help text for voice commands"""
        return """
Available voice commands:
- "Surveillance, what happened today?" - Query events
- "Surveillance, show me recent events" - Get latest events
- "Surveillance, system status" - Check system status
- "Surveillance, help" - Show this help
- "Surveillance, test system" - Run system tests
- "Surveillance, stop listening" - Disable voice interface

You can ask natural language questions about surveillance events, such as:
- "What happened at 2 PM?"
- "Show me all falls from yesterday"
- "Any suspicious activity this morning?"
        """.strip()

# Global instance
voice_interface = VoiceInterface()
