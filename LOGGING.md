# IPL Voice Assistant - Detailed Logging Guide

## Log Files Location

Logs are automatically created in: `/app/logs/voice_assistant_YYYYMMDD.log`

Example: `logs/voice_assistant_20251015.log`

## What Gets Logged

### 📞 Call Events
```
[call_abc123] 🎙️ Call started: call_abc123
[call_abc123] 📞 Caller: +919876543210
[call_abc123] ✅ First audio received after 2.3s
[call_abc123] 📴 Stop event received
[call_abc123] Call ended - Duration: 125.4s, Audio: True
```

### 🎤 Audio Processing (DEBUG level)
```
[call_abc123] 🎤 Received audio chunk: 1600 bytes at 8kHz
[call_abc123] 🔄 Resampled to 16kHz: 3200 bytes
[call_abc123] ✅ Sent 10 audio chunks to Gemini
[call_abc123] 📥 Received media event: 2144 chars base64, 1600 bytes raw
```

### 🤖 Gemini Responses
```
[call_abc123] 🔊 Audio data received from Gemini
[call_abc123] Decoded audio: 4800 bytes at 24kHz
[call_abc123] 🔄 Resampled to 8kHz: 1600 bytes
[call_abc123] ✅ Sent audio chunk #5 to Exotel
[call_abc123] 🤖 Assistant said: Great question! Which match are you asking about?
[call_abc123] 👤 User said: Who won yesterday's match?
[call_abc123] 🔄 Turn complete - Gemini finished speaking
```

### ⏱️ Timeouts & Errors
```
[call_abc123] ⏱️ Max call duration reached (1800s)
[call_abc123] ⏱️ Audio timeout - no audio for 90.5s
[call_abc123] ⏱️ No audio received for 30s, closing session
[call_abc123] ❌ Audio processing error: [error details with stack trace]
```

### 🔧 Session Management
```
[call_abc123] Starting Gemini session for call call_abc123
[call_abc123] ✅ Gemini session established for call call_abc123
[call_abc123] Starting audio send loop to Gemini
[call_abc123] Starting audio receive loop from Gemini
[call_abc123] Audio send loop ended. Total chunks sent: 45
[call_abc123] Audio receive loop ended
[call_abc123] Total responses: 3
[call_abc123] Session cleaned up. Active calls: 0
```

## Viewing Logs

### 1. Via Cloud Run Console
```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=ipl-voice-assistant" \
    --project=ai-voice-assistant-kots \
    --limit 100 \
    --format="table(timestamp,textPayload)"
```

### 2. Via API Endpoints

**Get recent logs (last 100 lines):**
```bash
curl https://your-service.run.app/logs
```

**Get last 500 lines:**
```bash
curl https://your-service.run.app/logs?lines=500
```

**Download full log file:**
```bash
curl https://your-service.run.app/logs/download -o voice_assistant.log
```

### 3. Real-time Streaming
```bash
gcloud logging tail "resource.type=cloud_run_revision AND resource.labels.service_name=ipl-voice-assistant" \
    --project=ai-voice-assistant-kots
```

### 4. Filter by Log Level
```bash
# Only errors
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=ipl-voice-assistant AND severity>=ERROR" \
    --project=ai-voice-assistant-kots --limit 50

# Only warnings and above
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=ipl-voice-assistant AND severity>=WARNING" \
    --project=ai-voice-assistant-kots --limit 50
```

### 5. Filter by Call ID
```bash
# Get logs for specific call
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=ipl-voice-assistant AND textPayload:call_abc123" \
    --project=ai-voice-assistant-kots
```

## Log Levels

| Level | Purpose | Logged To |
|-------|---------|-----------|
| **DEBUG** | Detailed audio chunks, every step | File only |
| **INFO** | Call events, user/bot speech, metrics | File + Console |
| **WARNING** | Timeouts, non-critical issues | File + Console |
| **ERROR** | Failures, exceptions with stack traces | File + Console |

## Typical Call Flow Logs

```
2025-10-15 10:30:15 - __main__ - [INFO] - exotel_stream_handler:356 - 📡 WebSocket connection established with Exotel
2025-10-15 10:30:15 - __main__ - [INFO] - exotel_stream_handler:381 - 🎙️ Call started: CAxxx
2025-10-15 10:30:15 - __main__ - [INFO] - exotel_stream_handler:382 - 📞 Caller: +919876543210
2025-10-15 10:30:15 - __main__ - [INFO] - start:197 - Starting Gemini session for call CAxxx
2025-10-15 10:30:16 - __main__ - [INFO] - start:217 - ✅ Gemini session established for call CAxxx
2025-10-15 10:30:17 - __main__ - [INFO] - add_audio:346 - [CAxxx] ✅ First audio received after 2.1s
2025-10-15 10:30:18 - __main__ - [INFO] - _receive_audio_from_gemini:317 - [CAxxx] 🤖 Assistant said: Hello! Welcome to IPL Voice Assistant!
2025-10-15 10:30:22 - __main__ - [INFO] - _receive_audio_from_gemini:327 - [CAxxx] 👤 User said: Who won yesterday's match?
2025-10-15 10:30:23 - __main__ - [INFO] - _receive_audio_from_gemini:317 - [CAxxx] 🤖 Assistant said: Great question! Which match are you asking about?
2025-10-15 10:30:30 - __main__ - [INFO] - exotel_stream_handler:509 - [CAxxx] 📴 Stop event received
2025-10-15 10:30:30 - __main__ - [INFO] - stop:358 - [CAxxx] Call ended - Duration: 15.2s, Audio: True
```

## Debugging Tips

### 1. No Audio Received
Look for:
- `⏱️ CONNECTION TIMEOUT: No audio received in 90 seconds`
- Check if Exotel is sending `start` event
- Verify WebSocket URL in Exotel config

### 2. Bot Not Responding
Look for:
- Gemini session establishment: `✅ Gemini session established`
- Audio chunks being sent: `✅ Sent N audio chunks to Gemini`
- Check for errors: `❌ Error receiving audio from Gemini`

### 3. Poor Audio Quality
Look for:
- Audio chunk sizes (should be consistent)
- Resampling logs (8kHz → 16kHz → 24kHz → 8kHz)
- Any audio processing errors

### 4. Call Dropped Unexpectedly
Look for:
- `⏱️ Audio timeout - no audio for 90.5s`
- `⏱️ Max call duration reached`
- WebSocket disconnect events

## Log Rotation

Logs are automatically rotated daily. Each day creates a new file:
- `voice_assistant_20251015.log`
- `voice_assistant_20251016.log`
- etc.

**Note:** Cloud Run instances are ephemeral, so logs are only persisted to Cloud Logging (accessible via gcloud commands).

## Production Recommendations

1. **Monitor ERROR logs** - Set up alerts for error rate spikes
2. **Track timeout_calls metric** - High timeouts = Exotel configuration issue
3. **Monitor average_duration** - Sudden changes = user behavior shift
4. **Check success_rate** - Should be >95%

## Support

If you see unusual patterns in logs, check:
1. Exotel dashboard for call connectivity
2. Gemini API quotas/limits
3. Cloud Run instance health
4. Network latency between Exotel and Cloud Run
