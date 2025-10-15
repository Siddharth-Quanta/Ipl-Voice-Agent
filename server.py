"""
IPL Voice Assistant - Cloud Run + Exotel Integration
Handles phone calls via Exotel WebSocket and Gemini Live API
"""

import os
import json
import base64
import asyncio
import logging
import time
from typing import Optional, Dict
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.responses import Response, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import websockets
from google import genai
from google.genai.types import LiveConnectConfig, PrebuiltVoiceConfig, VoiceConfig, SpeechConfig
import numpy as np
from pydub import AudioSegment
from io import BytesIO

# Configure detailed logging to both file and console
# Use Indian Standard Time (IST)
IST = ZoneInfo("Asia/Kolkata")
os.makedirs("logs", exist_ok=True)
log_file = f"logs/voice_assistant_{datetime.now(IST).strftime('%Y%m%d')}.log"

# Create formatters with IST timezone
class ISTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=IST)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat()

detailed_formatter = ISTFormatter(
    '%(asctime)s - %(name)s - [%(levelname)s] - %(funcName)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S IST'
)

# File handler - detailed logs
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(detailed_formatter)

# Console handler - less verbose with IST
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = ISTFormatter('%(asctime)s IST - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console_handler.setFormatter(console_formatter)

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[file_handler, console_handler]
)

logger = logging.getLogger(__name__)
logger.info(f"=" * 80)
logger.info(f"IPL Voice Assistant Starting - Logs: {log_file}")
logger.info(f"=" * 80)

# Initialize FastAPI
app = FastAPI(title="IPL Voice Assistant")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CLOUD_RUN_URL = os.getenv("CLOUD_RUN_URL", "")

# Validate API key
if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY not set!")
    raise ValueError("GOOGLE_API_KEY environment variable required")

# Audio configuration
EXOTEL_SAMPLE_RATE = 8000  # Exotel provides 8kHz
GEMINI_INPUT_RATE = 16000  # Gemini expects 16kHz
GEMINI_OUTPUT_RATE = 24000  # Gemini outputs 24kHz
CHUNK_SIZE = 3200  # 100ms at 16kHz

# Call management
MAX_CALL_DURATION = 1800  # 30 minutes max per call
AUDIO_TIMEOUT = 90  # 90 seconds without audio = disconnect
active_sessions: Dict[str, 'ExotelGeminiSession'] = {}

# Connection metrics
connection_metrics = {
    'total_calls': 0,
    'successful_calls': 0,
    'failed_calls': 0,
    'timeout_calls': 0,
    'active_calls': 0,
    'total_duration': 0,
    'longest_call': 0,
}

# Gemini client
client = genai.Client(api_key=GOOGLE_API_KEY, http_options={"api_version": "v1alpha"})


def _create_ipl_assistant_prompt() -> str:
    """System prompt for IPL voice assistant"""
    return """You are an enthusiastic IPL (Indian Premier League) cricket expert and voice assistant.

ROLE:
- Friendly, knowledgeable cricket enthusiast
- Provide real-time IPL match updates, stats, and insights
- Help users with team information, player stats, and match schedules

CONVERSATION STYLE:
- Keep responses SHORT (1-2 sentences max for voice)
- Be enthusiastic but concise
- Use cricket terminology naturally
- Speak in a conversational, friendly tone

TOPICS YOU HANDLE:
✅ Match scores and updates
✅ Team standings and rankings
✅ Player statistics and performance
✅ Match schedules and fixtures
✅ Team compositions and strategies
✅ Historical IPL data and records
✅ Fantasy cricket suggestions

IMPORTANT RULES:
- NEVER give long monologues - this is a VOICE conversation
- Ask clarifying questions if the user's request is unclear
- If you don't know current match data, acknowledge it honestly
- Keep the cricket spirit alive with your enthusiasm!

SAMPLE INTERACTIONS:
User: "Who won yesterday's match?"
You: "Great question! Which match are you asking about - CSK vs MI or RCB vs KKR?"

User: "Tell me about Virat Kohli's performance"
You: "Virat's having a stellar season! He's scored 450 runs with 4 fifties. What specific stats would you like?"

User: "Who should I pick for fantasy cricket?"
You: "For today's match, I'd suggest MS Dhoni as captain - he's in red-hot form! Need more picks?"

VOICE CONVERSATION TIPS:
- Confirm what you heard if uncertain: "Did you say Mumbai Indians?"
- Give options for follow-ups: "Want to know more about their batting or bowling?"
- React naturally: "Wow!", "That's incredible!", "Close match!"
- Keep the energy high - it's cricket! 🏏

Remember: You're speaking to cricket fans who want QUICK, EXCITING info - not reading a Wikipedia article!
"""


class AudioResampler:
    """Handle audio resampling between different sample rates"""

    @staticmethod
    def resample_audio(audio_data: bytes, from_rate: int, to_rate: int, channels: int = 1) -> bytes:
        """Resample audio from one rate to another"""
        try:
            # Convert bytes to AudioSegment
            audio = AudioSegment(
                data=audio_data,
                sample_width=2,  # 16-bit = 2 bytes
                frame_rate=from_rate,
                channels=channels
            )

            # Resample
            resampled = audio.set_frame_rate(to_rate)

            # Convert back to bytes
            return resampled.raw_data
        except Exception as e:
            logger.error(f"Error resampling audio: {e}")
            return audio_data


class ExotelGeminiSession:
    """Manages a single call session between Exotel and Gemini"""

    def __init__(self, call_sid: str, exotel_ws: WebSocket):
        self.call_sid = call_sid
        self.exotel_ws = exotel_ws
        self.gemini_session = None
        self.audio_queue = asyncio.Queue()
        self.is_active = True
        self.resampler = AudioResampler()
        self.start_time = time.time()
        self.last_audio_time = None
        self.audio_received = False
        self.caller_number = None

    async def start(self):
        """Start the Gemini Live session"""
        try:
            logger.info(f"Starting Gemini session for call {self.call_sid}")

            # Configure Gemini Live session with simple config
            config = LiveConnectConfig(
                response_modalities=["AUDIO"]
            )

            # Connect to Gemini
            async with client.aio.live.connect(
                model="models/gemini-2.0-flash-exp",
                config=config
            ) as session:
                self.gemini_session = session
                logger.info(f"✅ Gemini session established for call {self.call_sid}")

                # Send system instruction as initial message
                system_prompt = _create_ipl_assistant_prompt()
                await session.send(system_prompt, end_of_turn=True)
                logger.debug(f"[{self.call_sid}] System instruction sent to Gemini")

                # Run bidirectional audio streaming
                await asyncio.gather(
                    self._send_audio_to_gemini(),
                    self._receive_audio_from_gemini(),
                    return_exceptions=True
                )

        except Exception as e:
            logger.error(f"Error in Gemini session: {e}")
        finally:
            self.is_active = False
            logger.info(f"Gemini session ended for call {self.call_sid}")

    async def _send_audio_to_gemini(self):
        """Send audio from Exotel to Gemini (8kHz → 16kHz)"""
        audio_chunks_sent = 0
        try:
            logger.debug(f"[{self.call_sid}] Starting audio send loop to Gemini")

            while self.is_active:
                # Get audio from queue
                exotel_audio = await asyncio.wait_for(
                    self.audio_queue.get(),
                    timeout=30.0
                )

                if exotel_audio is None:
                    logger.debug(f"[{self.call_sid}] Received stop signal in audio send loop")
                    break

                # Log audio chunk details
                logger.debug(f"[{self.call_sid}] 🎤 Received audio chunk: {len(exotel_audio)} bytes at 8kHz")

                # Resample 8kHz → 16kHz for Gemini
                gemini_audio = self.resampler.resample_audio(
                    exotel_audio,
                    from_rate=EXOTEL_SAMPLE_RATE,
                    to_rate=GEMINI_INPUT_RATE
                )

                logger.debug(f"[{self.call_sid}] 🔄 Resampled to 16kHz: {len(gemini_audio)} bytes")

                # Send to Gemini
                await self.gemini_session.send(input=gemini_audio, end_of_turn=False)
                audio_chunks_sent += 1

                if audio_chunks_sent % 10 == 0:  # Log every 10 chunks
                    logger.info(f"[{self.call_sid}] ✅ Sent {audio_chunks_sent} audio chunks to Gemini")

            logger.info(f"[{self.call_sid}] Audio send loop ended. Total chunks sent: {audio_chunks_sent}")

        except asyncio.TimeoutError:
            logger.warning(f"[{self.call_sid}] ⏱️ No audio received for 30s, closing session")
        except Exception as e:
            logger.error(f"[{self.call_sid}] ❌ Error sending audio to Gemini: {e}", exc_info=True)

    async def _receive_audio_from_gemini(self):
        """Receive audio from Gemini and send to Exotel (24kHz → 8kHz)"""
        audio_chunks_received = 0
        text_responses = []

        try:
            logger.debug(f"[{self.call_sid}] Starting audio receive loop from Gemini")

            async for response in self.gemini_session.receive():
                logger.debug(f"[{self.call_sid}] 📨 Received response from Gemini")

                # Handle audio output
                if response.data:
                    logger.debug(f"[{self.call_sid}] 🔊 Audio data received from Gemini")

                    # Decode base64 audio from Gemini (24kHz PCM)
                    gemini_audio = base64.b64decode(response.data)
                    logger.debug(f"[{self.call_sid}] Decoded audio: {len(gemini_audio)} bytes at 24kHz")

                    # Resample 24kHz → 8kHz for Exotel
                    exotel_audio = self.resampler.resample_audio(
                        gemini_audio,
                        from_rate=GEMINI_OUTPUT_RATE,
                        to_rate=EXOTEL_SAMPLE_RATE
                    )

                    logger.debug(f"[{self.call_sid}] 🔄 Resampled to 8kHz: {len(exotel_audio)} bytes")

                    # Send to Exotel via WebSocket
                    await self.exotel_ws.send_json({
                        "event": "media",
                        "media": {
                            "payload": base64.b64encode(exotel_audio).decode('utf-8')
                        }
                    })

                    audio_chunks_received += 1
                    logger.debug(f"[{self.call_sid}] ✅ Sent audio chunk #{audio_chunks_received} to Exotel")

                # Handle text output (for logging transcriptions)
                if response.text:
                    logger.info(f"[{self.call_sid}] 🤖 Assistant said: {response.text}")
                    text_responses.append(response.text)

                # Handle user input transcription
                if hasattr(response, 'server_content') and response.server_content:
                    if hasattr(response.server_content, 'model_turn'):
                        model_turn = response.server_content.model_turn
                        if hasattr(model_turn, 'parts'):
                            for part in model_turn.parts:
                                if hasattr(part, 'text') and part.text:
                                    logger.info(f"[{self.call_sid}] 👤 User said: {part.text}")

                # Handle turn completion
                if response.server_content and response.server_content.turn_complete:
                    logger.info(f"[{self.call_sid}] 🔄 Turn complete - Gemini finished speaking")
                    logger.info(f"[{self.call_sid}] Total audio chunks sent: {audio_chunks_received}")

            logger.info(f"[{self.call_sid}] Audio receive loop ended")
            logger.info(f"[{self.call_sid}] Total responses: {len(text_responses)}")

        except Exception as e:
            logger.error(f"[{self.call_sid}] ❌ Error receiving audio from Gemini: {e}", exc_info=True)

    async def add_audio(self, audio_data: bytes):
        """Add incoming audio from Exotel to processing queue"""
        if not self.audio_received:
            self.audio_received = True
            duration = time.time() - self.start_time
            logger.info(f"[{self.call_sid}] ✅ First audio received after {duration:.1f}s")

        self.last_audio_time = time.time()
        await self.audio_queue.put(audio_data)

    async def stop(self):
        """Stop the session and update metrics"""
        self.is_active = False
        await self.audio_queue.put(None)

        # Calculate call duration
        duration = time.time() - self.start_time
        connection_metrics['total_duration'] += duration

        if duration > connection_metrics['longest_call']:
            connection_metrics['longest_call'] = duration

        if self.audio_received:
            connection_metrics['successful_calls'] += 1

        logger.info(f"[{self.call_sid}] Call ended - Duration: {duration:.1f}s, Audio: {self.audio_received}")


@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "IPL Voice Assistant",
        "timestamp": datetime.now(IST).isoformat(),
        "gemini_configured": bool(GOOGLE_API_KEY),
        "active_calls": len(active_sessions),
        "total_calls_handled": connection_metrics['total_calls'],
        "success_rate": round(
            (connection_metrics['successful_calls'] / connection_metrics['total_calls'] * 100)
            if connection_metrics['total_calls'] > 0 else 0,
            2
        )
    }


@app.post("/exotel/answer")
async def exotel_answer_webhook(request: Request):
    """
    Exotel calls this endpoint when a call is received
    Returns XML to configure call flow
    """
    form_data = await request.form()
    call_sid = form_data.get("CallSid", "unknown")
    from_number = form_data.get("From", "unknown")
    to_number = form_data.get("To", "unknown")

    logger.info(f"📞 Incoming call: {call_sid} from {from_number} to {to_number}")

    # Construct WebSocket URL for streaming
    ws_url = f"{CLOUD_RUN_URL.replace('https://', 'wss://')}/exotel/stream"

    # Return Exotel XML with Voicebot applet
    xml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>Welcome to IPL Voice Assistant! Connecting you now.</Say>
    <VoiceBot>
        <WebSocketUrl>{ws_url}</WebSocketUrl>
    </VoiceBot>
</Response>"""

    return Response(content=xml_response, media_type="application/xml")


@app.get("/exotel/stream-url")
async def get_stream_url():
    """Returns dynamic WebSocket URL for Exotel configuration"""
    ws_url = f"{CLOUD_RUN_URL.replace('https://', 'wss://')}/exotel/stream"
    return {"websocket_url": ws_url}


@app.websocket("/exotel/stream")
async def exotel_stream_handler(websocket: WebSocket):
    """
    WebSocket endpoint for Exotel voice streaming
    Handles bidirectional audio between Exotel and Gemini
    """
    call_sid = None
    session = None

    try:
        # Accept WebSocket connection
        await websocket.accept()
        logger.info("📡 WebSocket connection established with Exotel")

        # Update metrics
        connection_metrics['total_calls'] += 1
        connection_metrics['active_calls'] = len(active_sessions) + 1

        # Wait for connected or start event from Exotel
        first_message = None
        try:
            first_message = await asyncio.wait_for(
                websocket.receive_json(),
                timeout=10.0
            )
            logger.info(f"📨 First message from Exotel: {first_message.get('event')}")
        except asyncio.TimeoutError:
            logger.error("⏱️ Timeout waiting for initial event from Exotel")
            connection_metrics['failed_calls'] += 1
            await websocket.close(code=1008, reason="No initial event received")
            return

        # Handle 'connected' event - wait for 'start' event next
        if first_message.get("event") == "connected":
            logger.info("✅ Received 'connected' event, waiting for 'start' event...")
            try:
                start_message = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=10.0
                )
                logger.info(f"📨 Second message from Exotel: {start_message.get('event')}")
            except asyncio.TimeoutError:
                logger.error("⏱️ Timeout waiting for start event after connected")
                connection_metrics['failed_calls'] += 1
                await websocket.close(code=1008, reason="No start event after connected")
                return
        else:
            start_message = first_message

        if start_message.get("event") == "start":
            # Extract call information
            start_data = start_message.get("start", {})
            call_sid = start_data.get("callSid", f"call_{int(time.time())}")
            caller_number = start_data.get("from", "unknown")

            logger.info(f"🎙️ Call started: {call_sid}")
            logger.info(f"📞 Caller: {caller_number}")

            # Create and start Gemini session
            session = ExotelGeminiSession(call_sid, websocket)
            session.caller_number = caller_number
            active_sessions[call_sid] = session

            # Start Gemini task
            gemini_task = asyncio.create_task(session.start())

            # Process incoming messages from Exotel
            while session.is_active:
                try:
                    # Check timeouts
                    current_time = time.time()

                    # Max call duration check
                    if (current_time - session.start_time) > MAX_CALL_DURATION:
                        logger.warning(f"[{call_sid}] ⏱️ Max call duration reached ({MAX_CALL_DURATION}s)")
                        await websocket.close(code=1000, reason="Max duration reached")
                        break

                    # Audio timeout check
                    if session.audio_received and session.last_audio_time:
                        silence_duration = current_time - session.last_audio_time
                        if silence_duration > AUDIO_TIMEOUT:
                            logger.warning(f"[{call_sid}] ⏱️ Audio timeout - no audio for {silence_duration:.1f}s")
                            connection_metrics['timeout_calls'] += 1
                            await websocket.close(code=1008, reason="Audio timeout")
                            break

                    # Receive message with timeout
                    message = await asyncio.wait_for(
                        websocket.receive_json(),
                        timeout=5.0
                    )

                    event = message.get("event")

                    if event == "media":
                        # Extract and process audio payload
                        payload = message.get("media", {}).get("payload")
                        if payload:
                            try:
                                audio_data = base64.b64decode(payload)
                                logger.debug(f"[{call_sid}] 📥 Received media event: {len(payload)} chars base64, {len(audio_data)} bytes raw")
                                await session.add_audio(audio_data)
                            except Exception as audio_err:
                                logger.error(f"[{call_sid}] ❌ Audio processing error: {audio_err}", exc_info=True)

                    elif event == "stop":
                        logger.info(f"[{call_sid}] 📴 Stop event received")
                        break

                    elif event == "mark":
                        mark_name = message.get("mark", {}).get("name", "unknown")
                        logger.debug(f"[{call_sid}] Mark: {mark_name}")

                except asyncio.TimeoutError:
                    # Timeout on receive - check if session is still active
                    if not session.is_active:
                        break
                    continue

                except Exception as msg_err:
                    logger.error(f"[{call_sid}] Error processing message: {msg_err}")
                    break

            # Stop session
            if session:
                await session.stop()

            # Wait for Gemini task to complete (with timeout)
            try:
                await asyncio.wait_for(gemini_task, timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning(f"[{call_sid}] Timeout waiting for Gemini task to finish")
                gemini_task.cancel()

        else:
            logger.error(f"Expected 'start' event, got: {start_message.get('event')}")
            connection_metrics['failed_calls'] += 1
            await websocket.close(code=1002, reason="Invalid start event")

    except WebSocketDisconnect:
        logger.info(f"[{call_sid}] WebSocket disconnected")
        if not session or not session.audio_received:
            connection_metrics['failed_calls'] += 1

    except Exception as e:
        logger.error(f"[{call_sid}] WebSocket error: {e}", exc_info=True)
        connection_metrics['failed_calls'] += 1

    finally:
        # Cleanup
        if session:
            await session.stop()

        if call_sid and call_sid in active_sessions:
            del active_sessions[call_sid]

        connection_metrics['active_calls'] = len(active_sessions)
        logger.info(f"[{call_sid}] Session cleaned up. Active calls: {len(active_sessions)}")


@app.post("/exotel/passthru")
async def exotel_passthru(request: Request):
    """
    Exotel calls this after the Voicebot applet completes
    Used for logging and analytics
    """
    form_data = await request.form()
    call_sid = form_data.get("CallSid", "unknown")
    status = form_data.get("Status", "unknown")
    duration = form_data.get("Duration", "0")

    logger.info(f"📊 Call completed: {call_sid}, Status: {status}, Duration: {duration}s")

    return PlainTextResponse("OK")


@app.get("/logs")
async def get_recent_logs(lines: int = 100):
    """Get recent log lines from the log file"""
    try:
        if not os.path.exists(log_file):
            return {"error": "Log file not found", "path": log_file}

        with open(log_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines

        return {
            "log_file": log_file,
            "total_lines": len(all_lines),
            "showing_lines": len(recent_lines),
            "logs": [line.strip() for line in recent_lines]
        }
    except Exception as e:
        logger.error(f"Error reading log file: {e}")
        return {"error": str(e)}


@app.get("/logs/download")
async def download_logs():
    """Download the full log file"""
    try:
        if not os.path.exists(log_file):
            return {"error": "Log file not found"}

        from fastapi.responses import FileResponse
        return FileResponse(
            path=log_file,
            filename=f"voice_assistant_{datetime.now(IST).strftime('%Y%m%d_%H%M%S')}.log",
            media_type='text/plain'
        )
    except Exception as e:
        return {"error": str(e)}


@app.get("/stats")
async def get_stats():
    """Detailed stats endpoint with call metrics"""
    return {
        "service": "IPL Voice Assistant",
        "status": "operational",
        "timestamp": datetime.now(IST).isoformat(),
        "metrics": {
            "total_calls": connection_metrics['total_calls'],
            "successful_calls": connection_metrics['successful_calls'],
            "failed_calls": connection_metrics['failed_calls'],
            "timeout_calls": connection_metrics['timeout_calls'],
            "active_calls": connection_metrics['active_calls'],
            "total_duration_seconds": round(connection_metrics['total_duration'], 2),
            "longest_call_seconds": round(connection_metrics['longest_call'], 2),
            "average_duration_seconds": round(
                connection_metrics['total_duration'] / connection_metrics['successful_calls']
                if connection_metrics['successful_calls'] > 0 else 0,
                2
            ),
        },
        "active_sessions": [
            {
                "call_sid": sid,
                "caller": session.caller_number,
                "duration": round(time.time() - session.start_time, 2),
                "audio_received": session.audio_received,
            }
            for sid, session in active_sessions.items()
        ],
        "config": {
            "max_call_duration": MAX_CALL_DURATION,
            "audio_timeout": AUDIO_TIMEOUT,
            "gemini_model": "gemini-2.0-flash-exp",
        }
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
