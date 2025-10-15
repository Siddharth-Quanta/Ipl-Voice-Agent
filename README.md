# IPL Voice Assistant

AI-powered voice assistant for IPL (Indian Premier League) cricket queries using Google Cloud Run and Exotel integration.

## 🏏 Features

- Real-time voice interaction via phone calls
- IPL match information, player stats, and schedules
- Powered by Google Gemini 2.0 Flash (Live API)
- Low-latency audio streaming optimized for India
- Automatic call handling with Exotel integration
- Detailed logging and metrics tracking

## 🚀 Technology Stack

- **Backend**: FastAPI + Python 3.11
- **AI Model**: Google Gemini 2.0 Flash Live API
- **Telephony**: Exotel (India)
- **Deployment**: Google Cloud Run (asia-south1/Mumbai)
- **Audio Processing**: pydub, numpy

## 📋 Prerequisites

1. Google Cloud Platform account with billing enabled
2. Google Gemini API key ([Get one here](https://aistudio.google.com/apikey))
3. Exotel account with voice streaming enabled
4. Google Cloud SDK installed

## 🛠️ Setup

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd ipl-voice-assistant
```

### 2. Configure Environment Variables

Create `.env` file:

```bash
GOOGLE_API_KEY=your_gemini_api_key_here
CLOUD_RUN_URL=https://your-service.run.app
```

### 3. Deploy to Cloud Run

```bash
# Authenticate with Google Cloud
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable cloudbuild.googleapis.com run.googleapis.com

# Deploy
gcloud run deploy ipl-voice-assistant \
    --source . \
    --region asia-south1 \
    --allow-unauthenticated \
    --set-env-vars="GOOGLE_API_KEY=YOUR_KEY,CLOUD_RUN_URL=https://your-service.run.app"
```

### 4. Configure Exotel

1. Login to [Exotel Dashboard](https://my.exotel.com/)
2. Go to **App Bazaar** → **Create New App**
3. Add **Voicebot** applet
4. Set WebSocket URL: `wss://your-service.run.app/exotel/stream`
5. Connect to your phone number

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check with basic stats |
| `/stats` | GET | Detailed metrics (calls, duration, success rate) |
| `/logs?lines=N` | GET | Get recent N log lines |
| `/logs/download` | GET | Download full log file |
| `/exotel/answer` | POST | Exotel call initiation webhook |
| `/exotel/stream` | WebSocket | Audio streaming endpoint |

## 🔍 Monitoring

### View Logs

```bash
# Real-time logs
gcloud logging tail "resource.type=cloud_run_revision AND resource.labels.service_name=ipl-voice-assistant"

# Recent logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=ipl-voice-assistant" --limit 100
```

### Check Stats

```bash
curl https://your-service.run.app/stats
```

## 📝 Log Format

Logs include detailed information:

```
📡 WebSocket connection established
🎙️ Call started: call_xyz
📞 Caller: +919876543210
👤 User said: Who won yesterday's match?
🤖 Assistant said: Great question! Which match are you asking about?
📴 Stop event received
```

## 🎯 Configuration

### Call Limits

- **Max call duration**: 1800 seconds (30 minutes)
- **Audio timeout**: 90 seconds of silence
- **Sample rates**: 8kHz (Exotel) ↔ 16kHz (Gemini) ↔ 24kHz (output)

### Customization

Edit the system prompt in `server.py` (line ~79):

```python
def _create_ipl_assistant_prompt() -> str:
    return """You are an enthusiastic IPL cricket expert...
    # Customize your bot's personality and knowledge here
    """
```

## 💰 Cost Estimation

**For 120 calls/day @ 5 minutes average:**

- Cloud Run: ~$12-15/month
- Exotel voice: ~$30/month
- Gemini API: Free tier (15 RPM)
- **Total**: ~$42-45/month

Scales to zero when idle → $0 cost during inactive periods.

## 🐛 Troubleshooting

### No Audio Received

1. Verify Exotel voice streaming is enabled
2. Check WebSocket URL in Exotel config
3. Review logs for connection errors

### Gemini Errors

1. Check API key validity
2. Verify API quotas not exceeded
3. Check logs for detailed error messages

### High Latency

1. Ensure region is `asia-south1` (Mumbai)
2. Check Exotel server location
3. Monitor Cloud Run CPU usage

## 📚 Documentation

- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - Detailed deployment instructions
- [LOGGING.md](LOGGING.md) - Logging and debugging guide

## 🔐 Security

- Never commit `.env` file or actual API keys
- Use Google Secret Manager for production secrets
- Enable Cloud Run authentication for sensitive endpoints
- Regularly rotate API keys

## 📄 License

[Add your license here]

## 🤝 Contributing

[Add contribution guidelines if needed]

## 🙏 Acknowledgments

- Google Gemini for AI capabilities
- Exotel for telephony infrastructure
- Google Cloud Run for scalable deployment
