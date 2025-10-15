import os
import asyncio
import logging
import numpy as np
import time
import pyaudio
import threading
import collections
from enum import Enum
from typing import Optional, AsyncGenerator, Dict, Any
from google import genai
from google.genai.types import LiveConnectConfig, SpeechConfig, VoiceConfig, PrebuiltVoiceConfig, Blob, GenerationConfig
from dotenv import load_dotenv

# Import scipy for audio resampling
try:
    from scipy import signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print(" scipy not installed. Install with: pip install scipy")
    print("   Audio resampling will not be available\n")

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConversationState(Enum):
    """States for conversation management"""
    IDLE = "idle"
    LISTENING = "listening"
    RECORDING = "recording"
    PROCESSING = "processing"
    RESPONDING = "responding"

class IPLVoiceAgent:
    """IPL Voice Agent using Google Gemini Live API with voice input/output"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the IPL Voice Agent"""
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided or set in GOOGLE_API_KEY environment variable")

        # Initialize Gemini client
        self.client = genai.Client(
            api_key=self.api_key,
            http_options={'api_version': 'v1alpha'}
        )

        # Session state
        self.session = None
        self._session_context = None
        self.is_connected = False
        self.is_processing = False
        self.state = ConversationState.IDLE

        # Using Gemini's built-in VAD and interruption handling
        # Audio is streamed directly to Gemini which handles speech detection
        self.speech_started = False
        self.is_speaking = False  # Flag to track when agent is speaking (for UI only)
        self.playback_active = False  # Track active playback for interruption

        # Audio configuration - matching reference implementation
        self.INPUT_RATE = 16000   # 16kHz input required by Gemini
        self.OUTPUT_RATE = 24000  # 24kHz output from Gemini
        self.CHUNK_DURATION_MS = 100  # 100ms chunks for smoother streaming

        # Transcription and audio buffers (like reference)
        self.input_transcription_buffer = ""
        self.output_transcription_buffer = ""
        self.input_audio_buffer = b''
        self.output_audio_buffer = b''

        # IPL system prompt
        self.system_prompt = self._create_ipl_system_prompt()

        logger.info("IPL Voice Agent initialized with Gemini Live API")

    def _create_ipl_system_prompt(self) -> str:
        """Create IPL-specific system prompt for voice interaction"""
        return """You are an IPL-ONLY cricket voice assistant. You MUST STRICTLY discuss ONLY IPL (Indian Premier League) topics. Respond in a friendly, conversational tone suitable for voice interaction.

IMPORTANT LANGUAGE INSTRUCTIONS:
- The user will speak in INDIAN ENGLISH (with Indian accent and pronunciation)
- You MUST interpret ALL input as English, transcribe with English characters
- Indian English may include Hindi/regional language words mixed with English - interpret them in IPL context
- Always respond ONLY in English
- Be familiar with Indian cricket terminology and player names

Your expertise is LIMITED TO:
- IPL teams ONLY: CSK, MI, RCB, KKR, DC, RR, PBKS, GT, LSG, SRH
- IPL player statistics and performances (2008-present)
- IPL auctions, retentions, and transfers
- IPL match results, venues, and schedules
- IPL records, winners, and award winners
- IPL-specific rules and regulations

ABSOLUTE RESTRICTIONS:
- DO NOT answer about international cricket (World Cup, Test matches, ODIs, T20Is)
- DO NOT answer about other cricket leagues (BBL, PSL, CPL, The Hundred, etc.)
- DO NOT answer about non-cricket topics
- DO NOT answer about cricket that is not IPL-related

MANDATORY RESPONSE for non-IPL questions:
"I can only answer questions about IPL. Please ask me about IPL teams, players, matches, auctions, or statistics."

VOICE INTERACTION GUIDELINES:
- Keep responses concise but informative (2-3 sentences max)
- Use natural, conversational language
- Speak clearly and at a moderate pace
- For numbers and statistics, speak them clearly
- If someone asks about World Cup or other topics, politely redirect to IPL
- Stay STRICTLY within IPL boundaries. No exceptions."""

    def _create_config(self) -> LiveConnectConfig:
        """Create the Live API configuration with transcription enabled"""
        config = LiveConnectConfig(
            response_modalities=["AUDIO"],  # Voice output
            system_instruction=self.system_prompt,
            input_audio_transcription={},  # Enable input transcription
            output_audio_transcription={},  # Enable output transcription
            generation_config=GenerationConfig(
                temperature=0.3,  # Consistent responses
                max_output_tokens=150,  # Concise voice responses
                candidate_count=1
            )
        )

        # Add speech configuration for Indian English
        config.speech_config = SpeechConfig(
            voice_config=VoiceConfig(
                prebuilt_voice_config=PrebuiltVoiceConfig(
                    voice_name="Charon"  # Clear voice that works well with Indian accent
                )
            ),
            language_code="en-IN"  # Indian English for better accent recognition
        )

        return config

    async def start_session(self) -> bool:
        """Start a Gemini Live session"""
        try:
            logger.info("Starting IPL Voice Agent session...")
            config = self._create_config()

            # Create session context
            self._session_context = self.client.aio.live.connect(
                model="gemini-2.0-flash-live-001",
                config=config
            )

            # Start the session
            self.session = await asyncio.wait_for(
                self._session_context.__aenter__(),
                timeout=30.0
            )

            self.is_connected = True
            self.state = ConversationState.LISTENING
            logger.info("IPL Voice Agent session started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start session: {e}")
            self.is_connected = False
            return False

    async def end_session(self):
        """End the Gemini Live session"""
        try:
            self.is_connected = False
            self.is_processing = False
            self.state = ConversationState.IDLE

            if self._session_context and self.session:
                await asyncio.wait_for(
                    self._session_context.__aexit__(None, None, None),
                    timeout=5.0
                )

            self.session = None
            self._session_context = None
            logger.info("IPL Voice Agent session ended")

        except Exception as e:
            logger.error(f"Error ending session: {e}")

    def _resample_audio(self, audio_data: bytes, input_rate: int, output_rate: int) -> bytes:
        """Resample audio from one sample rate to another"""
        if not HAS_SCIPY:
            return audio_data  # Return original if scipy not available

        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)

            # Calculate resampling ratio
            resample_ratio = output_rate / input_rate

            # Calculate output length
            output_length = int(len(audio_array) * resample_ratio)

            # Use scipy's resample for high-quality resampling
            resampled = signal.resample(audio_array, output_length)

            # Convert back to int16
            resampled = np.clip(resampled, -32768, 32767).astype(np.int16)

            # Convert to bytes
            return resampled.tobytes()

        except Exception as e:
            logger.error(f"Error resampling audio: {e}")
            return audio_data  # Return original if resampling fails

    async def send_audio(self, audio_data: bytes) -> bool:
        """Send audio data to Gemini Live"""
        if not self.is_connected or not self.session:
            return False

        try:
            # Resample if needed (from 16kHz to 16kHz - no change needed for Gemini)
            # Gemini accepts 16kHz PCM audio directly

            # Create audio blob with proper format
            audio_blob = Blob(
                mime_type="audio/pcm",
                data=audio_data
            )

            # Send to Gemini - it handles VAD and transcription
            await self.session.send_realtime_input(audio=audio_blob)

            # Add to input buffer for transcription tracking
            self.input_audio_buffer += audio_data

            return True

        except Exception as e:
            if "keepalive" not in str(e) and "1011" not in str(e):
                logger.error(f"Error sending audio: {e}")
            return False

    async def receive_responses(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Receive responses from Gemini Live with transcription support"""
        if not self.is_connected or not self.session:
            logger.error("Cannot receive responses - not connected")
            return

        try:
            async for response in self.session.receive():
                # Handle different response types
                if hasattr(response, 'server_content') and response.server_content:
                    server_content = response.server_content

                    # Handle input audio transcription (what user said)
                    if hasattr(server_content, 'input_transcription') and server_content.input_transcription:
                        transcript = server_content.input_transcription
                        if hasattr(transcript, 'text') and transcript.text:
                            logger.info(f"User said: {transcript.text}")
                            self.input_transcription_buffer += transcript.text
                            yield {
                                'type': 'user_speech',
                                'data': transcript.text
                            }

                    # Handle output audio transcription (what model said)
                    if hasattr(server_content, 'output_transcription') and server_content.output_transcription:
                        transcript = server_content.output_transcription
                        if hasattr(transcript, 'text') and transcript.text:
                            # Buffer the transcription instead of logging each fragment
                            self.output_transcription_buffer += transcript.text
                            # Don't yield individual fragments to avoid clutter

                    # Handle user turn (legacy)
                    if hasattr(server_content, 'user_turn') and server_content.user_turn:
                        if hasattr(server_content.user_turn, 'parts') and server_content.user_turn.parts:
                            for part in server_content.user_turn.parts:
                                if hasattr(part, 'text') and part.text:
                                    logger.info(f"User said: {part.text}")
                                    yield {
                                        'type': 'user_speech',
                                        'data': part.text
                                    }

                    # Handle model response with audio
                    if hasattr(server_content, 'model_turn') and server_content.model_turn:
                        if hasattr(server_content.model_turn, 'parts') and server_content.model_turn.parts:
                            for part in server_content.model_turn.parts:
                                # Handle audio response - buffer it
                                if hasattr(part, 'inline_data') and part.inline_data:
                                    if hasattr(part.inline_data, 'data') and part.inline_data.data:
                                        # Buffer audio instead of yielding immediately
                                        self.output_audio_buffer += part.inline_data.data

                                # Handle text response (for logging)
                                if hasattr(part, 'text') and part.text:
                                    logger.info(f"IPL Agent response: {part.text}")
                                    yield {
                                        'type': 'text',
                                        'data': part.text
                                    }


                    # Handle interruptions - Gemini signals when user interrupts
                    if hasattr(server_content, 'interrupted') and server_content.interrupted:
                        logger.info("User interrupted agent - stopping playback")
                        yield {
                            'type': 'interruption',
                            'data': 'User interrupted the agent'
                        }
                        # Clear output buffers on interruption
                        self.output_transcription_buffer = ""
                        self.output_audio_buffer = b''

                    # Handle turn completion
                    if hasattr(server_content, 'turn_complete') and server_content.turn_complete:
                        # Yield complete user input if we have it
                        if self.input_transcription_buffer:
                            logger.info(f"User complete: {self.input_transcription_buffer}")
                            yield {
                                'type': 'user_speech_complete',
                                'data': self.input_transcription_buffer
                            }

                        # Yield the complete audio response after all chunks are received
                        if self.output_audio_buffer:
                            logger.info(f"Playing complete audio response ({len(self.output_audio_buffer)} bytes)")
                            yield {
                                'type': 'audio_complete',
                                'data': self.output_audio_buffer
                            }

                        # Yield the complete model response transcription if we have it
                        if self.output_transcription_buffer:
                            logger.info(f"Model complete response: {self.output_transcription_buffer}")
                            yield {
                                'type': 'model_speech',
                                'data': self.output_transcription_buffer
                            }

                        # Clear transcription buffers on turn complete
                        self.input_transcription_buffer = ""
                        self.output_transcription_buffer = ""
                        self.input_audio_buffer = b''
                        self.output_audio_buffer = b''
                        yield {
                            'type': 'turn_complete',
                            'data': None
                        }

        except Exception as e:
            logger.error(f"Error receiving responses: {e}")
            yield {
                'type': 'error',
                'data': str(e)
            }

    async def voice_conversation(self, audio_input_queue: asyncio.Queue, audio_output_queue: asyncio.Queue, audio_handler=None):
        """Main voice conversation loop with automatic reconnection and interruption support"""
        reconnect_attempts = 0
        max_reconnects = 5

        while reconnect_attempts < max_reconnects:
            try:
                if not await self.start_session():
                    logger.error("Failed to start session")
                    reconnect_attempts += 1
                    await asyncio.sleep(2)
                    continue

                self.is_processing = True
                reconnect_attempts = 0  # Reset on successful connection
                logger.info("Session active, listening...")

                # Create concurrent tasks for input and output
                tasks = [
                    asyncio.create_task(self._handle_audio_input(audio_input_queue)),
                    asyncio.create_task(self._handle_audio_output(audio_output_queue, audio_handler))
                ]

                # Wait for tasks to complete
                await asyncio.gather(*tasks, return_exceptions=True)

            except asyncio.CancelledError:
                logger.info("Conversation cancelled")
                break
            except Exception as e:
                error_msg = str(e)
                if "keepalive ping timeout" in error_msg or "1011" in error_msg:
                    logger.info("Connection timeout, reconnecting...")
                    reconnect_attempts += 1
                    await asyncio.sleep(1)
                else:
                    logger.error(f"Error in conversation: {e}")
                    break
            finally:
                # Clean up current session
                self.is_processing = False
                for task in locals().get('tasks', []):
                    if not task.done():
                        task.cancel()
                await self.end_session()

        if reconnect_attempts >= max_reconnects:
            logger.error("Max reconnection attempts reached")

    async def _handle_audio_input(self, audio_input_queue: asyncio.Queue):
        """Stream audio continuously to Gemini - let it handle VAD"""
        logger.info("Audio streaming started - Gemini VAD is active")
        chunks_processed = 0

        while self.is_processing and self.is_connected:
            try:
                # Get audio data with timeout
                audio_data = await asyncio.wait_for(audio_input_queue.get(), timeout=0.5)

                if audio_data is None:  # Stop signal
                    break

                # Stream directly to Gemini - it will handle VAD and interruptions
                # Note: Echo cancellation must be handled at audio hardware/OS level
                if self.session:
                    await self.send_audio(audio_data)

                    chunks_processed += 1
                    if chunks_processed % 50 == 0:  # Log every ~5 seconds
                        logger.info(f"Still streaming... {chunks_processed} audio chunks sent")

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                if "keepalive ping timeout" in str(e) or "1011" in str(e):
                    logger.info("Connection lost in input handler")
                    raise  # Propagate to trigger reconnection
                logger.error(f"Error handling audio input: {e}")
                await asyncio.sleep(0.1)

    async def _handle_audio_output(self, audio_output_queue: asyncio.Queue, audio_handler=None):
        """Handle audio output from Gemini with interruption support"""
        while self.is_processing and self.is_connected:
            try:
                async for response in self.receive_responses():
                    if not self.is_processing:
                        break

                    if response['type'] == 'interruption':
                        # User interrupted - stop playback immediately
                        print("\n ⚠️  Interrupted by user...")
                        if audio_handler:
                            audio_handler.stop_playback()

                    elif response['type'] == 'audio_complete' and response['data']:
                        # Play the complete audio response after all chunks are received
                        try:
                            print(" Playing agent response...")
                            await asyncio.wait_for(
                                audio_output_queue.put(response['data']),
                                timeout=10.0  # Increased timeout for complete audio
                            )
                        except asyncio.TimeoutError:
                            logger.warning("Timeout putting complete audio in output queue")

                    elif response['type'] == 'user_speech_complete':
                        # Display complete user utterance
                        print(f"\n You said: {response['data']}")

                    elif response['type'] == 'user_speech':
                        # Skip individual fragments
                        pass

                    elif response['type'] == 'model_speech':
                        # Display complete model response transcription
                        print(f" Agent: {response['data']}")

                    elif response['type'] == 'text':
                        # Text response (if no transcription available)
                        print(f" IPL Agent: {response['data']}")

                    elif response['type'] == 'turn_complete':
                        print("\n Ready for next question...\n")
                        # Continue listening for next turn
                        continue

                    elif response['type'] == 'error':
                        logger.error(f"Response error: {response['data']}")
                        if "keepalive" in str(response['data']) or "1011" in str(response['data']):
                            raise Exception("Connection lost")  # Trigger reconnection
                        await asyncio.sleep(1)  # Brief pause before retrying

            except Exception as e:
                if not self.is_processing:
                    break
                logger.error(f"Error in output handler: {e}")
                if "keepalive" in str(e) or "1011" in str(e):
                    raise  # Propagate to trigger reconnection
                await asyncio.sleep(1)  # Wait before retrying


class AudioHandler:
    """Handle microphone input and speaker output with echo cancellation"""

    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.recording = False
        self.playing = False

        # Audio configuration matching reference (100ms chunks)
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.INPUT_RATE = 16000  # 16kHz for Gemini input
        self.OUTPUT_RATE = 24000  # 24kHz for Gemini output
        self.CHUNK_DURATION_MS = 100  # 100ms chunks for better streaming
        # 100ms at 16kHz = 1600 samples, 2 bytes per sample = 3200 bytes
        self.CHUNK = 1600  # 100ms at 16kHz

        # Echo cancellation: Track recent playback audio for subtraction
        self.playback_buffer = collections.deque(maxlen=50)  # Last 5 seconds of playback
        self.enable_software_aec = True  # Software-based echo cancellation
        self.aec_threshold = 0.3  # Threshold for echo detection
        self.current_playback_stream = None  # Track current playback stream for interruption

    def _apply_echo_cancellation(self, input_audio: bytes) -> bytes:
        """Apply simple acoustic echo cancellation to remove agent's voice from input"""
        if not self.enable_software_aec or len(self.playback_buffer) == 0:
            return input_audio

        try:
            # Convert input to numpy array
            input_array = np.frombuffer(input_audio, dtype=np.int16).astype(np.float32)

            # Simple energy-based suppression when agent is playing
            # If playback buffer has recent audio, reduce input gain
            if len(self.playback_buffer) > 0:
                # Calculate energy ratio
                input_energy = np.sqrt(np.mean(input_array ** 2))

                # Get most recent playback chunk
                recent_playback = self.playback_buffer[-1] if self.playback_buffer else None

                if recent_playback and len(recent_playback) > 0:
                    playback_array = np.frombuffer(recent_playback, dtype=np.int16).astype(np.float32)

                    # Resample playback from 24kHz to 16kHz for comparison
                    if HAS_SCIPY and len(playback_array) > 0:
                        target_len = int(len(playback_array) * 16000 / 24000)
                        playback_resampled = signal.resample(playback_array, target_len)

                        # Ensure matching lengths
                        min_len = min(len(input_array), len(playback_resampled))
                        if min_len > 0:
                            input_segment = input_array[:min_len]
                            playback_segment = playback_resampled[:min_len]

                            playback_energy = np.sqrt(np.mean(playback_segment ** 2))

                            # If input energy is similar to playback, it's likely echo
                            if playback_energy > 100 and input_energy > 100:  # Avoid division by zero
                                similarity = min(input_energy, playback_energy) / max(input_energy, playback_energy)

                                if similarity > self.aec_threshold:
                                    # Reduce input by 70% when echo detected
                                    input_array = input_array * 0.3
                                    logger.debug(f"Echo detected (similarity: {similarity:.2f}), applying suppression")

            # Convert back to int16
            output_array = np.clip(input_array, -32768, 32767).astype(np.int16)
            return output_array.tobytes()

        except Exception as e:
            logger.error(f"Error in echo cancellation: {e}")
            return input_audio

    def start_recording(self, audio_queue: asyncio.Queue, loop):
        """Start recording from microphone in a separate thread"""
        def record():
            try:
                # List all available microphones
                mic_devices = []
                for i in range(self.audio.get_device_count()):
                    info = self.audio.get_device_info_by_index(i)
                    if info['maxInputChannels'] > 0:
                        mic_devices.append((i, info['name']))
                        print(f"Found microphone [{i}]: {info['name']}")

                # Use the first non-mapper device if available
                mic_index = None
                for idx, name in mic_devices:
                    if "Microsoft Sound Mapper" not in name:
                        mic_index = idx
                        print(f"Selected microphone: {name}")
                        break

                # Fallback to first available if no specific device found
                if mic_index is None and mic_devices:
                    mic_index = mic_devices[0][0]
                    print(f"Using default microphone: {mic_devices[0][1]}")

                if mic_index is None:
                    print(" No microphone found!")

                stream = self.audio.open(
                    format=self.FORMAT,
                    channels=self.CHANNELS,
                    rate=self.INPUT_RATE,
                    input=True,
                    input_device_index=mic_index,
                    frames_per_buffer=self.CHUNK
                )

                print("🎤 Recording started. Speak about IPL...")
                self.recording = True

                while self.recording:
                    try:
                        data = stream.read(self.CHUNK, exception_on_overflow=False)

                        # Apply echo cancellation to remove agent's voice
                        cleaned_data = self._apply_echo_cancellation(data)

                        # Put cleaned audio data in queue using the provided loop
                        asyncio.run_coroutine_threadsafe(
                            audio_queue.put(cleaned_data),
                            loop
                        )
                    except Exception as e:
                        logger.error(f"Error reading audio: {e}")
                        break

                stream.stop_stream()
                stream.close()
                print(" Recording stopped")

            except Exception as e:
                logger.error(f"Error in recording: {e}")
                print(f" Recording failed: {e}")

        # Start recording in background thread
        recording_thread = threading.Thread(target=record)
        recording_thread.daemon = True
        recording_thread.start()

    def stop_recording(self):
        """Stop recording"""
        self.recording = False

    def play_audio(self, audio_data: bytes, allow_interruption: bool = True):
        """Play audio data through speakers with interruption support"""
        try:
            logger.info(f"Playing audio: {len(audio_data)} bytes at {self.OUTPUT_RATE}Hz")
            # Create a stream for playback (24kHz from Gemini)
            stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.OUTPUT_RATE,  # Gemini outputs at 24kHz
                output=True,
                frames_per_buffer=4096  # Larger buffer for smoother playback
            )

            self.current_playback_stream = stream
            self.playing = True

            # Play the audio in chunks for better control and interruption
            chunk_size = 4096
            for i in range(0, len(audio_data), chunk_size):
                # Check if playback was interrupted
                if allow_interruption and not self.playing:
                    logger.info("Playback interrupted by user")
                    break

                chunk = audio_data[i:i+chunk_size]
                stream.write(chunk)

                # Add chunk to playback buffer for echo cancellation
                self.playback_buffer.append(chunk)

            self.playing = False
            self.current_playback_stream = None
            stream.stop_stream()
            stream.close()
            logger.info("Audio playback completed")

        except Exception as e:
            self.playing = False
            self.current_playback_stream = None
            logger.error(f"Error playing audio: {e}")

    def stop_playback(self):
        """Stop current audio playback (for interruption)"""
        try:
            self.playing = False
            if self.current_playback_stream:
                self.current_playback_stream.stop_stream()
                self.current_playback_stream.close()
                self.current_playback_stream = None
                logger.info("Playback stopped due to interruption")
        except Exception as e:
            logger.error(f"Error stopping playback: {e}")

    async def handle_audio_output(self, audio_output_queue: asyncio.Queue, agent=None):
        """Handle audio output from the queue with interruption support"""
        while True:
            try:
                # Get audio data with timeout
                audio_data = await asyncio.wait_for(audio_output_queue.get(), timeout=0.1)

                if audio_data is None:  # Stop signal
                    break

                # Set speaking flag for UI tracking (but don't mute microphone)
                if agent:
                    agent.is_speaking = True
                    agent.playback_active = True
                    logger.info("Agent started speaking - echo cancellation active")

                # Play audio in a thread but wait for completion
                # Audio can be interrupted by user speech (Gemini handles this)
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.play_audio, audio_data)

                # Clear speaking flag AFTER playback completes or is interrupted
                if agent:
                    agent.is_speaking = False
                    agent.playback_active = False
                    logger.info("Agent finished speaking")

                    # Clear playback buffer gradually (keep some for echo cancellation)
                    # Keep last 10 chunks for lingering echo
                    while len(self.playback_buffer) > 10:
                        self.playback_buffer.popleft()

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                if agent:
                    agent.is_speaking = False
                    agent.playback_active = False
                logger.error(f"Error handling audio output: {e}")

    def cleanup(self):
        """Cleanup audio resources"""
        self.recording = False
        self.playing = False
        try:
            self.audio.terminate()
        except:
            pass


async def main():
    """Main function to run the IPL Voice Agent"""
    audio_handler = None
    agent = None

    try:
        print(" IPL Voice Agent with Gemini Live (with Speech Detection)")
        print("=" * 50)

        print(" Using Gemini's built-in Voice Activity Detection\n")
        print(" Optimized for Indian English accent and pronunciation\n")

        print("Make sure your microphone and speakers are working!")
        print("\nYou can ask about:")
        print("- IPL teams (CSK, MI, RCB, KKR, DC, RR, PBKS, GT, LSG, SRH)")
        print("- Player statistics and performances")
        print("- Match results and schedules")
        print("- IPL auctions and records")
        print("\n Using Gemini's built-in Voice Activity Detection!")
        print(" Just speak naturally - Gemini will detect when you're talking")
        print("Press Enter to start, Ctrl+C to stop")
        input("\nReady? Press Enter...")

        # Initialize components
        agent = IPLVoiceAgent()
        audio_handler = AudioHandler()

        # Create audio queues
        audio_input_queue = asyncio.Queue()
        audio_output_queue = asyncio.Queue()

        # Get the current event loop for the thread
        loop = asyncio.get_event_loop()

        # Start audio input recording with the loop
        audio_handler.start_recording(audio_input_queue, loop)

        # Start audio output handler with agent reference
        output_task = asyncio.create_task(
            audio_handler.handle_audio_output(audio_output_queue, agent)
        )

        print("\n Streaming audio to Gemini...")
        print(" Gemini VAD is listening for your speech")
        print(" Responses will be spoken aloud")
        print("\n" + "="*50)

        # Start the voice conversation with audio handler for interruption support
        conversation_task = asyncio.create_task(
            agent.voice_conversation(audio_input_queue, audio_output_queue, audio_handler)
        )

        # Wait for conversation or user interruption
        await conversation_task

    except KeyboardInterrupt:
        print("\n Stopping IPL Voice Agent...")

    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f" Error: {e}")

    finally:
        # Cleanup
        if audio_handler:
            audio_handler.stop_recording()
            # Send stop signals to queues
            try:
                await audio_input_queue.put(None)
                await audio_output_queue.put(None)
            except:
                pass
            audio_handler.cleanup()

        if agent:
            await agent.end_session()

        print("Cleanup completed. Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())