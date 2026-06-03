import json
import base64
import asyncio
import logging
import websockets
import boto3
from groq import AsyncGroq
from server.config import config
from server.audio_utils import pcm_to_mulaw

logger = logging.getLogger(__name__)

# Initialize AWS Polly Client
polly_client = boto3.client(
    "polly",
    aws_access_key_id=config.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
    region_name=config.AWS_REGION
)

# Initialize Groq Client
groq_client = AsyncGroq(api_key=config.GROQ_API_KEY)


class VoiceConnectionManager:
    def __init__(self, twilio_ws):
        self.twilio_ws = twilio_ws
        self.stream_sid = None
        self.deepgram_ws = None
        self.transcription_buffer = ""
        self.is_bot_speaking = False
        
        # Load prompt
        prompt_path = config.PROMPTS_DIR / "investment_bot_prompt.txt"
        self.system_prompt = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else "You are a helpful assistant."
        self.chat_history = [{"role": "system", "content": self.system_prompt}]

    async def connect_to_deepgram(self):
        """Establish WebSocket connection to Deepgram for live STT."""
        deepgram_url = "wss://api.deepgram.com/v1/listen?encoding=mulaw&sample_rate=8000&channels=1&language=hi&model=nova-2&endpointing=300"
        
        headers = {
            "Authorization": f"Token {config.DEEPGRAM_API_KEY}"
        }
        
        try:
            self.deepgram_ws = await websockets.connect(deepgram_url, additional_headers=headers)
            logger.info("Connected to Deepgram STT.")
            # Start background task to receive Deepgram transcripts
            asyncio.create_task(self.receive_from_deepgram())
        except Exception as e:
            logger.error(f"Failed to connect to Deepgram: {e}")

    async def receive_from_deepgram(self):
        """Listen for transcripts from Deepgram."""
        if not self.deepgram_ws:
            return
            
        try:
            async for message in self.deepgram_ws:
                data = json.loads(message)
                logger.info(f"DG Raw: {message[:100]}")
                
                # Check for transcript results
                if data.get("type") == "Results":
                    channel = data.get("channel", {})
                    alts = channel.get("alternatives", [])
                    if alts:
                        transcript = alts[0].get("transcript", "").strip()
                        
                        # Trigger response only when user finishes a sentence (endpointing)
                        if transcript and (data.get("is_final") or data.get("speech_final")):
                            logger.info(f"User: {transcript}")
                            if not self.is_bot_speaking:
                                await self.process_llm_and_speak(transcript)
        except Exception as e:
            logger.error(f"Deepgram receive error: {e}")

    async def process_llm_and_speak(self, user_text: str):
        """Pass user text to Groq, then to Polly, then back to Twilio."""
        self.is_bot_speaking = True
        self.chat_history.append({"role": "user", "content": user_text})
        
        try:
            # 1. Get LLM Response
            response = await groq_client.chat.completions.create(
                model=config.LLM_MODEL or "llama-3.3-70b-versatile",
                messages=self.chat_history,
                temperature=0.7,
                max_tokens=100
            )
            bot_text = response.choices[0].message.content.strip()
            self.chat_history.append({"role": "assistant", "content": bot_text})
            logger.info(f"Bot: {bot_text}")

            # 2. Convert Text to Speech using AWS Polly
            polly_response = polly_client.synthesize_speech(
                Text=bot_text,
                OutputFormat="pcm",
                SampleRate="8000",
                VoiceId="Kajal",
                LanguageCode="hi-IN",
                Engine="neural"
            )
            
            # Read PCM audio stream from Polly
            audio_stream = polly_response.get("AudioStream")
            if audio_stream:
                pcm_data = audio_stream.read()
                
                # 3. Convert PCM to mulaw base64
                mulaw_b64 = pcm_to_mulaw(pcm_data)
                
                # 4. Send to Twilio
                payload = {
                    "event": "media",
                    "streamSid": self.stream_sid,
                    "media": {
                        "payload": mulaw_b64
                    }
                }
                await self.twilio_ws.send_json(payload)
                
        except Exception as e:
            logger.error(f"Error in LLM/TTS pipeline: {e}")
        finally:
            self.is_bot_speaking = False

    async def handle_twilio_message(self, message: str):
        """Handle incoming messages from Twilio WebSocket."""
        data = json.loads(message)
        event = data.get("event")
        
        if event == "connected":
            logger.info("Twilio WebSocket connected.")
        elif event == "start":
            self.stream_sid = data.get("start", {}).get("streamSid")
            logger.info(f"Stream started. SID: {self.stream_sid}")
            await self.connect_to_deepgram()
            
            # Trigger welcome message
            await self.process_llm_and_speak("Hello! Please introduce yourself warmly as Riya.")
            
        elif event == "media":
            # Pass audio to Deepgram
            if self.deepgram_ws and not self.is_bot_speaking:
                payload = data.get("media", {}).get("payload")
                if payload:
                    audio_bytes = base64.b64decode(payload)
                    await self.deepgram_ws.send(audio_bytes)
                    
        elif event == "stop":
            logger.info("Stream stopped.")
            if self.deepgram_ws:
                await self.deepgram_ws.close()
