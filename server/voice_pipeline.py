import re
import json
import base64
import asyncio
import logging
from pathlib import Path
from datetime import datetime
import websockets
import boto3
from groq import AsyncGroq
from server.config import config
from server.audio_utils import pcm_to_mulaw, chunk_pcm
from server.voice_state_machine import CallState, VoiceStateMachine, classify_permission, classify_language, extract_datetime, classify_bot_type

logger = logging.getLogger(__name__)

# Initialize AWS Polly Client
polly_client = boto3.client(
    "polly",
    aws_access_key_id=config.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
    region_name=config.AWS_REGION
)

# Initialize Groq Client with persistent connection pool
# Without this, each turn creates a new TCP+TLS connection (~4-8s on home WiFi).
# With this, the first call opens the connection and subsequent calls reuse it (~0ms).
import httpx as _httpx
_groq_http_client = _httpx.AsyncClient(
    limits=_httpx.Limits(
        max_connections=5,
        max_keepalive_connections=2,
        keepalive_expiry=120,       # Keep connection alive for 2 minutes between turns
    ),
    timeout=_httpx.Timeout(30.0, connect=10.0),  # 10s connect, 30s total
)
groq_client = AsyncGroq(api_key=config.GROQ_API_KEY, http_client=_groq_http_client)


class VoiceConnectionManager:
    def __init__(self, twilio_ws):
        self.twilio_ws = twilio_ws
        self.stream_sid = None
        self.deepgram_ws = None
        self.transcription_buffer = ""
        self.is_bot_speaking = False
        self.call_ended = False
        self.welcome_interrupted = False  # Barge-in flag for welcome message
        self.barge_in_transcript = ""
        self._deepgram_reconnect_count = 0
        self.last_speak_finished_time = 0.0
        self.agent_voice = "Kajal"
        self.agent_lang = "hi-IN"
        self.customer_category = ""
        
        self.state_machine = None

    async def _determine_bot_persona(self, start_data: dict):
        """
        Dynamically determine and load the correct system prompt, agent name, and voice
        based on the target client's phone number or category.
        """
        # In Twilio, the metadata is inside start_data.customParameters or we look up caller/callee details
        # Exotel has 'to' and 'from' at start. Twilio customParameters can be passed, or we look up by stream parameters.
        # Twilio starts have: 'customParameters', 'to', 'from', etc.
        phone = start_data.get("to") or start_data.get("from") or ""
        custom_params = start_data.get("customParameters", {})
        
        # If bot_type is directly passed as a parameter (e.g. from WhatsApp API call initiation)
        bot_type = custom_params.get("bot_type") or "investment"
        customer_name = custom_params.get("customer_name") or ""
        customer_category = custom_params.get("customer_category") or custom_params.get("category") or ""
        
        clean_phone = phone.replace("+", "").replace(" ", "").replace("-", "").strip()
        
        # If no explicit parameter was passed, lookup by phone number in scraped leads database
        leads_file = Path("data/leads/hni_leads_pune.csv")
        if leads_file.exists() and clean_phone and not customer_name:
            try:
                import csv
                with open(leads_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        row_phone = row.get("phone", "").replace("+", "").replace(" ", "").replace("-", "").strip()
                        if row_phone and (row_phone in clean_phone or clean_phone in row_phone):
                            customer_name = row.get("name", "").strip()
                            customer_category = row.get("category", "").strip()
                            break
            except Exception as e:
                logger.error(f"Error reading leads file: {e}")
                
        # Perform LLM classification on category if we have a category
        if customer_category:
            bot_type = await classify_bot_type(customer_category)
            logger.info(f"Classified lead category '{customer_category}' -> Bot Type: {bot_type}")

        # Set agent voice and language
        self.agent_voice = "Kajal"
        self.agent_lang = "hi-IN"
            
        self.bot_type = bot_type
        self.customer_name = customer_name
        self.caller_phone = clean_phone
        self.customer_category = customer_category
        self.state_machine = VoiceStateMachine(
            bot_type=bot_type, 
            customer_name=customer_name, 
            customer_category=customer_category
        )
        self.crm_context = {"profession": "", "city": "Pune"}
        logger.info(f"Active Persona State Machine | Voice: {self.agent_voice} | Target Client: {customer_name} | Category: {customer_category}")

    async def connect_to_deepgram(self):
        """Establish WebSocket connection to Deepgram for live STT."""
        deepgram_url = "wss://api.deepgram.com/v1/listen?encoding=mulaw&sample_rate=8000&channels=1&language=hi&model=nova-2&endpointing=300"
        
        headers = {
            "Authorization": f"Token {config.DEEPGRAM_API_KEY}"
        }
        
        try:
            self.deepgram_ws = await websockets.connect(deepgram_url, additional_headers=headers)
            self._deepgram_reconnect_count = 0
            logger.info("Connected to Deepgram STT.")
            # Start background task to receive Deepgram transcripts
            asyncio.create_task(self.receive_from_deepgram())
            # Start keepalive task
            asyncio.create_task(self.deepgram_keepalive())
        except Exception as e:
            logger.error(f"Failed to connect to Deepgram: {e}")

    async def deepgram_keepalive(self):
        """Send periodic KeepAlive messages to Deepgram to prevent timeout."""
        try:
            while self.deepgram_ws:
                await asyncio.sleep(5)
                if self.deepgram_ws:
                    logger.debug("Sending KeepAlive to Deepgram")
                    await self.deepgram_ws.send(json.dumps({"type": "KeepAlive"}))
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in Deepgram KeepAlive: {e}")

    async def receive_from_deepgram(self):
        """Listen for transcripts from Deepgram."""
        if not self.deepgram_ws:
            return
            
        try:
            async for message in self.deepgram_ws:
                data = json.loads(message)
                logger.debug(f"DG Raw: {message[:100]}")
                
                # Check for transcript results
                if data.get("type") == "Results":
                    channel = data.get("channel", {})
                    alts = channel.get("alternatives", [])
                    if alts:
                        transcript = alts[0].get("transcript", "").strip()
                        
                        # Clean punctuation
                        clean_transcript = re.sub(r'^[.,\s!?]+|[.,\s!?]+$', '', transcript).strip()
                        
                        # Trigger response only when user finishes a sentence (endpointing)
                        if clean_transcript and (data.get("is_final") or data.get("speech_final")):
                            logger.info(f"User: {clean_transcript}")
                            
                            now = asyncio.get_event_loop().time()
                            cooldown_elapsed = now - self.last_speak_finished_time
                            
                            if self.is_bot_speaking:
                                if getattr(self, 'state_machine', None) and self.state_machine.state == CallState.HANGUP:
                                    logger.info(f"Ignoring barge-in '{clean_transcript}' because call is in final goodbye.")
                                    continue
                                
                                # Barge-in: if user speaks during welcome/bot, interrupt it
                                self.welcome_interrupted = True
                                self.barge_in_transcript = clean_transcript
                                logger.info(f"User barged in with '{clean_transcript}' — interrupting bot.")
                                continue
                                
                            if cooldown_elapsed < 1.2:
                                logger.info(f"Ignoring transcript '{clean_transcript}' due to echo cooldown ({cooldown_elapsed:.2f}s < 1.2s).")
                                continue
                                
                            self.is_bot_speaking = True
                            asyncio.create_task(self.process_llm_and_speak(clean_transcript))
        except Exception as e:
            logger.error(f"Deepgram receive error: {e}")
        finally:
            self.deepgram_ws = None
            if not self.call_ended and self._deepgram_reconnect_count < 2:
                self._deepgram_reconnect_count += 1
                logger.warning(f"Deepgram connection dropped — reconnect attempt {self._deepgram_reconnect_count}/2")
                try:
                    await self.connect_to_deepgram()
                except Exception as e:
                    logger.error(f"Deepgram reconnect failed: {e}")
            elif not self.call_ended:
                logger.error("Deepgram reconnect limit reached — bot is deaf for remainder of call")

    async def process_llm_and_speak(self, user_text: str):
        """Pass user text to Groq, then to Polly, then back to Twilio."""
        if self.call_ended:
            return
        self.is_bot_speaking = True
        
        if not hasattr(self, 'state_machine') or self.state_machine is None:
            self.state_machine = VoiceStateMachine(
                bot_type=getattr(self, 'bot_type', 'investment'), 
                customer_name=getattr(self, 'customer_name', ''), 
                customer_category=getattr(self, 'customer_category', '')
            )

        try:
            # 1. Update history with the user input
            self.state_machine.chat_history.append({"role": "user", "content": user_text})

            # Keep history manageable — system prompt + last 10 messages
            if len(self.state_machine.chat_history) > 12:
                self.state_machine.chat_history = (
                    [self.state_machine.chat_history[0]]  # system prompt
                    + self.state_machine.chat_history[-10:]  # recent context
                )

            instruction = self.state_machine.get_instruction_for_current_state(user_text=user_text)
            
            call_is_ending = False

            if self.state_machine.state == CallState.HANGUP and instruction.startswith("The call has gone on too long"):
                bot_text = "माफ़ कीजिएगा, यह call काफी लंबी हो गई है। मैं आपको बाद में कॉल करूँगी। [CALL_END]"
                self.state_machine.chat_history.append({"role": "assistant", "content": bot_text})
                logger.info(f"Bot (Max Turns): {bot_text}")
                call_is_ending = True
            else:
                messages = list(self.state_machine.chat_history)
                messages.append({"role": "system", "content": instruction})

                # 2. Get LLM Response from Groq (with 1 retry on transient errors)
                bot_text = None
                for attempt in range(2):
                    try:
                        response = await groq_client.chat.completions.create(
                            model=config.LLM_MODEL or "llama-3.3-70b-versatile",
                            messages=messages,
                            temperature=0.3,
                            max_tokens=150
                        )
                        bot_text = response.choices[0].message.content.strip()
                        break  # Success
                    except Exception as llm_err:
                        if attempt == 0:
                            logger.warning(f"LLM attempt 1 failed: {llm_err}. Retrying in 1s...")
                            await asyncio.sleep(1.0)
                        else:
                            raise  # Let outer except handle it

                self.state_machine.chat_history.append({
                    "role": "assistant",
                    "content": bot_text
                })
                logger.info(f"Bot: {bot_text}")

                # Post-process: let state machine parse tags and transition
                self.state_machine.post_process_response(bot_text)

                if "[CALL_END]" in bot_text or self.state_machine.state == CallState.HANGUP:
                    call_is_ending = True
                    self.state_machine.state = CallState.HANGUP
            
            speak_text = re.sub(r'\[(?:CALL[\s_]*END|END[\s_]*CALL)\]', '', bot_text, flags=re.IGNORECASE).strip()
            speak_text = speak_text.replace("[HANG_UP]", "").strip()
            speak_text = re.sub(r'\[APPOINTMENT:.*?\]', '', speak_text, flags=re.IGNORECASE).strip()
            speak_text = re.sub(r'\[LEAD:.*?\]', '', speak_text, flags=re.IGNORECASE).strip()

            if speak_text:
                # Convert Text to Speech using AWS Polly
                is_ssml = speak_text.strip().startswith("<speak>") and speak_text.strip().endswith("</speak>")
                if is_ssml:
                    polly_text = speak_text
                else:
                    # Auto-wrap in SSML with natural pauses for LLM output
                    ssml_text = speak_text
                    ssml_text = ssml_text.replace("...", '<break time="350ms"/>')
                    ssml_text = ssml_text.replace("।", '।<break time="250ms"/>')
                    polly_text = f"<speak>{ssml_text}</speak>"

                polly_response = polly_client.synthesize_speech(
                    Text=polly_text,
                    TextType="ssml",
                    OutputFormat="pcm",
                    SampleRate="8000",
                    VoiceId=self.agent_voice,
                    LanguageCode=self.agent_lang,
                    Engine="neural"
                )
                
                # Read PCM audio stream from Polly
                audio_stream = polly_response.get("AudioStream")
                if audio_stream:
                    pcm_data = audio_stream.read()
                    
                    # Convert PCM to mulaw base64
                    mulaw_b64 = pcm_to_mulaw(pcm_data)
                    
                    # Send to Twilio
                    payload = {
                        "event": "media",
                        "streamSid": self.stream_sid,
                        "media": {
                            "payload": mulaw_b64
                        }
                    }
                    await self.twilio_ws.send_json(payload)
                    
            if call_is_ending:
                logger.info(f"Twilio Call Ending triggered — scoring lead and closing socket.")
                await self._score_and_log_lead()
                await asyncio.sleep(2.0)
                self.call_ended = True
                await self.twilio_ws.close()
                
        except Exception as e:
            logger.error(f"Error in LLM/TTS pipeline: {e}")
        finally:
            self.last_speak_finished_time = 0.0 if self.welcome_interrupted else asyncio.get_event_loop().time()
            self.is_bot_speaking = False
            
            if hasattr(self, 'barge_in_transcript') and self.barge_in_transcript and not self.call_ended:
                transcript_to_process = self.barge_in_transcript
                self.barge_in_transcript = ""
                asyncio.create_task(self.process_llm_and_speak(transcript_to_process))
                
            self.welcome_interrupted = False

    async def _score_and_log_lead(self):
        """Extract conversation signals, log the scored lead to CRM, and save full transcript (Twilio)."""
        try:
            from server.lead_scoring import LeadData, BotType, LeadSource, score_lead
            from server.sheets_manager import sheets_manager, whatsapp_notifier

            if not self.state_machine:
                return

            transcript_lines = []
            for m in self.state_machine.chat_history:
                if m['role'] in ('user', 'assistant'):
                    role = m['role']
                    content = m['content']
                    if role == 'assistant':
                        try:
                            parsed = json.loads(content)
                            content = parsed.get("response") or parsed.get("reply") or parsed.get("text") or content
                        except Exception:
                            pass
                    transcript_lines.append(f"{role}: {content}")
            transcript = "\n".join(transcript_lines)

            scheduled_day = self.state_machine.scheduled_day
            scheduled_time = self.state_machine.scheduled_time

            if self.state_machine.scheduled_day and self.state_machine.scheduled_time:
                outcome = "callback_scheduled"
            elif self.state_machine.scheduled_day:
                outcome = "callback_agreed"
            elif self.state_machine.state == CallState.HANGUP:
                outcome = "not_interested"
            else:
                outcome = "incomplete"

            bot_type_map = {"investment": BotType.INVESTMENT, "insurance": BotType.INSURANCE}
            lead = LeadData(
                name=getattr(self, 'customer_name', ''),
                phone=getattr(self, 'caller_phone', ''),
                conversation_summary=transcript[:500],
                source=LeadSource.OUTBOUND_CALL,
                bot_type=bot_type_map.get(getattr(self, 'bot_type', 'investment'), BotType.INVESTMENT),
                asked_for_callback=(self.state_machine.state == CallState.CONFIRM),
                ready_to_buy=(self.state_machine.state == CallState.CONFIRM),
                said_not_interested=(outcome == "not_interested"),
            )
            lead = score_lead(lead)

            # Save full transcript to disk
            try:
                call_log_dir = Path("data/call_logs")
                call_log_dir.mkdir(parents=True, exist_ok=True)
                call_log = {
                    "timestamp": datetime.now().isoformat(),
                    "phone": getattr(self, 'caller_phone', ''),
                    "customer_name": getattr(self, 'customer_name', ''),
                    "customer_category": getattr(self, 'customer_category', ''),
                    "bot_type": getattr(self, 'bot_type', 'investment'),
                    "outcome": outcome,
                    "scheduled_day": scheduled_day,
                    "scheduled_time": scheduled_time,
                    "lead_score": lead.score,
                    "lead_category": lead.category.value,
                    "full_transcript": transcript,
                }
                log_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{getattr(self, 'caller_phone', 'unknown')[-4:]}.json"
                log_path = call_log_dir / log_filename
                log_path.write_text(json.dumps(call_log, indent=2, ensure_ascii=False), encoding="utf-8")
                logger.info(f"[Twilio] Call transcript saved: {log_path}")
            except Exception as e:
                logger.error(f"[Twilio] Failed to save transcript to disk: {e}")

            if lead.category.value == "HOT":
                sheets_manager.log_hot_lead(lead, scheduled_day=scheduled_day, scheduled_time=scheduled_time)
                await whatsapp_notifier.notify_manager_hot_lead(lead, scheduled_day=scheduled_day, scheduled_time=scheduled_time)
                logger.info(f"[Twilio] 🔴 HOT LEAD logged: {lead.name or 'Unknown'} (Score: {lead.score})")
            elif lead.category.value == "WARM":
                sheets_manager.log_nurture_lead(lead)
                logger.info(f"[Twilio] 🟡 WARM LEAD logged: {lead.name or 'Unknown'} (Score: {lead.score})")
            else:
                logger.info(f"[Twilio] ⚪ COLD/DNC lead: Score {lead.score}")
        except Exception as e:
            logger.error(f"[Twilio] Error scoring/logging lead: {e}")

    async def play_welcome_message(self, customer_name: str, bot_type: str):
        """Play the opening welcome message immediately without calling Groq first, reducing latency.
        Uses SSML for natural pauses between phrases.
        """
        if not hasattr(self, 'state_machine') or self.state_machine is None:
            self.state_machine = VoiceStateMachine(
                bot_type=bot_type, 
                customer_name=customer_name, 
                customer_category=getattr(self, 'customer_category', '')
            )

        def generate_welcome_message(b_type: str, c_name: str, crm_context: dict = None) -> str:
            ctx = crm_context or {}
            profession    = ctx.get("profession", "")
            business_name = ctx.get("business_name", "")
            city          = ctx.get("city", "Pune")
            gender        = ctx.get("gender", "M")

            # ── Salutation ──────────────────────────────────
            if business_name:
                greeting_line = f'Namaste, <break time="150ms"/> क्या यह {business_name} है?'
            else:
                _BUSINESS_SIGNALS = {
                    "pvt", "ltd", "llp", "inc", "corp", "limited",
                    "clinic", "hospital", "pharmacy", "medical", "dental", "lab",
                    "dr", "dr.", "doctor", "centre", "center",
                    "enterprise", "enterprises", "traders", "trading", "agency",
                    "store", "shop", "mart", "school", "college", "institute", "foundation", "trust",
                    "associates", "works", "care", "studio"
                }
                name_lower = c_name.lower() if c_name else ""
                tokens = name_lower.replace(".", " ").split()
                is_business = bool(set(tokens) & _BUSINESS_SIGNALS)
                if is_business:
                    greeting_line = f'Namaste, <break time="150ms"/> क्या यह {c_name} है?'
                else:
                    greeting_line = f'Namaste {c_name} ji? <break time="150ms"/>' if c_name else f'Namaste? <break time="150ms"/>'

            # Name Verification Check: Just return the greeting, don't include the hook.
            # We add a slight interrogative pause so they respond.
            return f'<speak>{greeting_line}</speak>'

        welcome_text = generate_welcome_message(bot_type, customer_name, getattr(self, "crm_context", {}))
                
        self.state_machine.state = CallState.VERIFY_NAME
        # Clean SSML tags for chat history
        clean_welcome = welcome_text
        clean_welcome = re.sub(r'<[^>]+>', '', clean_welcome)
        clean_welcome = re.sub(r'\s+', ' ', clean_welcome).strip()
        self.state_machine.chat_history.append({"role": "assistant", "content": clean_welcome})
        
        logger.info(f"Welcoming customer directly: {welcome_text}")
        
        self.is_bot_speaking = True
        try:
            polly_response = polly_client.synthesize_speech(
                Text=welcome_text,
                TextType="ssml",
                OutputFormat="pcm",
                SampleRate="8000",
                VoiceId=self.agent_voice,
                LanguageCode=self.agent_lang,
                Engine="neural"
            )
            audio_stream = polly_response.get("AudioStream")
            if audio_stream:
                pcm_data = audio_stream.read()
                # Send in chunks to allow barge-in detection
                chunk_size = 640  # 640 bytes = 40ms at 8kHz 16-bit
                for i in range(0, len(pcm_data), chunk_size):
                    if self.welcome_interrupted or self.call_ended:
                        logger.info("Welcome message interrupted by user speech.")
                        break
                    chunk = pcm_data[i:i + chunk_size]
                    mulaw_b64 = pcm_to_mulaw(chunk)
                    payload = {
                        "event": "media",
                        "streamSid": self.stream_sid,
                        "media": {
                            "payload": mulaw_b64
                        }
                    }
                    await self.twilio_ws.send_json(payload)
                    await asyncio.sleep(0.04)  # 40ms pacing
        except Exception as e:
            logger.error(f"Error playing welcome: {e}")
        finally:
            self.last_speak_finished_time = 0.0 if self.welcome_interrupted else asyncio.get_event_loop().time()
            self.is_bot_speaking = False
            
            if hasattr(self, 'barge_in_transcript') and self.barge_in_transcript and not self.call_ended:
                transcript_to_process = self.barge_in_transcript
                self.barge_in_transcript = ""
                asyncio.create_task(self.process_llm_and_speak(transcript_to_process))
                
            self.welcome_interrupted = False

    async def handle_twilio_message(self, message: str):
        """Handle incoming messages from Twilio WebSocket."""
        data = json.loads(message)
        event = data.get("event")
        
        if event == "connected":
            logger.info("Twilio WebSocket connected.")
        elif event == "start":
            start_payload = data.get("start", {})
            self.stream_sid = start_payload.get("streamSid")
            logger.info(f"Stream started. SID: {self.stream_sid} | Payload: {data}")
            
            # Dynamically set persona
            await self._determine_bot_persona(start_payload)
            
            await self.connect_to_deepgram()
            
            # Trigger welcome message directly in background (bypass Groq)
            customer_name = getattr(self, 'customer_name', '')
            bot_type = getattr(self, 'bot_type', 'investment')
            asyncio.create_task(self.play_welcome_message(customer_name, bot_type))
            
        elif event == "media":
            # Pass audio to Deepgram
            if self.deepgram_ws:
                payload = data.get("media", {}).get("payload")
                if payload:
                    try:
                        audio_bytes = base64.b64decode(payload)
                        await self.deepgram_ws.send(audio_bytes)
                    except websockets.exceptions.ConnectionClosed as e:
                        logger.warning(f"Deepgram connection closed while sending media: {e}")
                        self.deepgram_ws = None
                    except Exception as e:
                        logger.error(f"Error sending media to Deepgram: {e}")
                    
        elif event == "stop":
            logger.info("Stream stopped.")
            # Score and log lead if not already done (e.g. caller hung up)
            if not self.call_ended and self.state_machine:
                await self._score_and_log_lead()
                self.call_ended = True
            if self.deepgram_ws:
                try:
                    await self.deepgram_ws.close()
                except Exception:
                    pass
                self.deepgram_ws = None
            try:
                await self.twilio_ws.close()
            except Exception:
                pass


class ExotelVoiceConnectionManager:
    """
    Handles bidirectional audio streaming between Exotel and the AI pipeline.
    Exotel streams 16-bit Linear PCM (not mu-law like Twilio), so we can
    skip the pcm_to_mulaw conversion and stream raw PCM directly.

    Features:
    - [CALL_END] tag detection: LLM appends this when conversation should end
    - Silence watchdog: Hangs up if no speech for 25 seconds
    - Graceful hangup: Closes WebSocket cleanly after final audio plays
    """

    def __init__(self, exotel_ws):
        self.exotel_ws = exotel_ws
        self.stream_sid = None
        self.deepgram_ws = None
        self.transcription_buffer = ""
        self.is_bot_speaking = False
        self.call_ended = False  # Flag to prevent processing after hangup
        self.welcome_interrupted = False  # Barge-in flag for welcome message
        self.barge_in_transcript = ""
        self._deepgram_reconnect_count = 0
        self.silence_watchdog_task = None  # Tracks the silence timer
        self.agent_voice = "Kajal"
        self.agent_lang = "hi-IN"
        self.outbound_chunk_index = 1
        self.outbound_timestamp_ms = 0
        self.last_speak_finished_time = 0.0
        self.customer_category = ""

        self.state_machine = None

    async def _determine_bot_persona(self, start_data: dict):
        """
        Dynamically determine and load the correct system prompt, agent name, and voice
        based on the target client's phone number or category.
        """
        phone = start_data.get("to") or start_data.get("from") or ""
        custom_params = start_data.get("customParameters", {}) or start_data.get("custom_parameters", {})
        
        bot_type = custom_params.get("bot_type") or "investment"
        customer_name = custom_params.get("customer_name") or ""
        customer_category = custom_params.get("customer_category") or custom_params.get("category") or ""
        
        def normalize_to_10_digits(phone_str: str) -> str:
            # Remove all non-numeric characters
            nums = re.sub(r'\D', '', phone_str)
            # If it starts with 91 and has 11+ digits, it contains a country code + optional leading zero
            if nums.startswith('91') and len(nums) > 10:
                nums = nums[2:]
            # Strip any leading zeros
            nums = nums.lstrip('0')
            # Return last 10 digits
            return nums[-10:] if len(nums) >= 10 else nums

        clean_phone = normalize_to_10_digits(phone)
        
        # If no explicit parameter was passed, lookup by phone number in scraped leads database
        leads_file = Path("data/leads/hni_leads_pune.csv")
        if leads_file.exists() and clean_phone and not customer_name:
            try:
                import csv
                with open(leads_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        row_phone = normalize_to_10_digits(row.get("phone", ""))
                        if row_phone and row_phone == clean_phone:
                            customer_name = row.get("name", "").strip()
                            customer_category = row.get("category", "").strip()
                            break
            except Exception as e:
                logger.error(f"[Exotel] Error reading leads file: {e}")
                
        # --- CRM Context Extraction for Welcome Message ---
        CATEGORY_MAP = {
            "doctors in pune":               {"profession": "doctor",  "city": "Pune", "bot_type": "insurance"},
            "architects in pune":            {"profession": "architect","city": "Pune", "bot_type": "recruitment"},
            "chartered accountants":         {"profession": "CA",      "city": "Pune", "bot_type": "recruitment"},
            "salaried professionals":        {"profession": "",        "city": "Pune", "bot_type": "investment"},
            "investment":                    {"profession": "",        "city": "Pune", "bot_type": "investment"},
            "insurance":                     {"profession": "",        "city": "Pune", "bot_type": "insurance"},
            "reminder":                      {"profession": "",        "city": "Pune", "bot_type": "reminder"}
        }
        
        self.crm_context = CATEGORY_MAP.get(customer_category.lower().strip(), {"profession": "", "city": "Pune"})
        
        # Bypass LLM classification if we already mapped the bot_type
        mapped_bot_type = CATEGORY_MAP.get(customer_category.lower().strip(), {}).get("bot_type")
        if mapped_bot_type:
            bot_type = mapped_bot_type
        elif customer_category:
            # Only use LLM if it's an unknown category
            bot_type = await classify_bot_type(customer_category)
            logger.info(f"[Exotel] Classified lead category '{customer_category}' -> Bot Type: {bot_type}")

        # Set agent voice and language
        self.agent_voice = "Kajal"
        self.agent_lang = "hi-IN"
            
        self.bot_type = bot_type
        self.customer_name = customer_name
        self.customer_category = customer_category
        self.caller_phone = clean_phone  # Store for CRM logging at hangup
            
        # Infer business name if the customer name sounds like a business (simple heuristic)
        if any(w in customer_name.lower() for w in ["clinic", "hospital", "associates", "enterprises", "solutions"]):
            self.crm_context["business_name"] = customer_name
            self.customer_name = ""  # Clear the name so we greet the business directly
            
        self.state_machine = VoiceStateMachine(
            bot_type=bot_type, 
            customer_name=self.customer_name, 
            customer_category=customer_category
        )
        logger.info(f"[Exotel] Active Persona State Machine | Bot Type: {bot_type} | CRM Context: {self.crm_context} | Voice: {self.agent_voice} | Target Client: {self.customer_name} | Phone: {clean_phone} | Category: {customer_category}")

    async def connect_to_deepgram(self):
        """Establish WebSocket connection to Deepgram for live STT.
        Uses linear16 encoding because Exotel streams raw PCM (not mu-law).
        """
        deepgram_url = "wss://api.deepgram.com/v1/listen?encoding=linear16&sample_rate=8000&channels=1&language=hi&model=nova-2&endpointing=300"

        headers = {
            "Authorization": f"Token {config.DEEPGRAM_API_KEY}"
        }

        try:
            self.deepgram_ws = await websockets.connect(deepgram_url, additional_headers=headers)
            self._deepgram_reconnect_count = 0
            logger.info("[Exotel] Connected to Deepgram STT (linear16).")
            asyncio.create_task(self.receive_from_deepgram())
            # Start keepalive task
            asyncio.create_task(self.deepgram_keepalive())
            # Start silence watchdog
            self._reset_silence_watchdog()
        except Exception as e:
            logger.error(f"[Exotel] Failed to connect to Deepgram: {e}")

    async def deepgram_keepalive(self):
        """Send periodic KeepAlive messages to Deepgram to prevent timeout."""
        try:
            while self.deepgram_ws and not self.call_ended:
                await asyncio.sleep(5)
                if self.deepgram_ws and not self.call_ended:
                    logger.debug("[Exotel] Sending KeepAlive to Deepgram")
                    await self.deepgram_ws.send(json.dumps({"type": "KeepAlive"}))
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[Exotel] Error in Deepgram KeepAlive: {e}")

    def _reset_silence_watchdog(self):
        """Reset the silence watchdog timer. Called every time the user speaks."""
        if self.silence_watchdog_task and not self.silence_watchdog_task.done():
            self.silence_watchdog_task.cancel()
        self.silence_watchdog_task = asyncio.create_task(self._silence_watchdog())

    async def _silence_watchdog(self):
        """Hang up if no speech detected for too long.
        Uses generous timeouts to avoid cutting off natural pauses.
        """
        try:
            # Wait 45 seconds of complete silence before prompting
            await asyncio.sleep(45)
            if self.call_ended:
                return

            # Ask if they're still there — Devanagari for correct Polly pronunciation
            logger.info("[Exotel] Silence detected (45s). Prompting user...")
            await self._speak_text(
                '<speak>Hello? <break time="400ms"/> आप सुन रहे हैं?</speak>'
            )

            # Wait 15 more seconds for a response
            await asyncio.sleep(15)
            if self.call_ended:
                return

            # Still no response — hang up
            logger.info("[Exotel] No response after silence prompt. Hanging up.")
            await self._speak_text(
                '<speak>लगता है connection में कोई issue है। <break time="300ms"/> '
                'मैं बाद में call करूँगी। <break time="200ms"/> नमस्ते!</speak>'
            )
            await asyncio.sleep(3)  # Let the audio play
            await self._hangup()

        except asyncio.CancelledError:
            pass  # Timer was reset because user spoke — this is normal

    async def receive_from_deepgram(self):
        """Listen for transcripts from Deepgram."""
        if not self.deepgram_ws:
            return

        try:
            async for message in self.deepgram_ws:
                if self.call_ended:
                    break

                data = json.loads(message)
                logger.debug(f"[Exotel] DG Raw: {message[:100]}")

                if data.get("type") == "Results":
                    channel = data.get("channel", {})
                    alts = channel.get("alternatives", [])
                    if alts:
                        transcript = alts[0].get("transcript", "").strip()

                        # Clean punctuation
                        clean_transcript = re.sub(r'^[.,\s!?]+|[.,\s!?]+$', '', transcript).strip()

                        if clean_transcript and (data.get("is_final") or data.get("speech_final")):
                            logger.info(f"[Exotel] User: {clean_transcript}")
                            # Reset silence watchdog — user is speaking
                            self._reset_silence_watchdog()
                            
                            now = asyncio.get_event_loop().time()
                            cooldown_elapsed = now - self.last_speak_finished_time
                            
                            if self.is_bot_speaking:
                                if getattr(self, 'state_machine', None) and self.state_machine.state == CallState.HANGUP:
                                    logger.info(f"[Exotel] Ignoring barge-in '{clean_transcript}' because call is in final goodbye.")
                                    continue
                                
                                # Barge-in: if user speaks during welcome/bot, interrupt it
                                self.welcome_interrupted = True
                                self.barge_in_transcript = clean_transcript
                                logger.info(f"[Exotel] User barged in with '{clean_transcript}' — interrupting bot.")
                                continue
                                
                            if cooldown_elapsed < 1.2:
                                logger.info(f"[Exotel] Ignoring transcript '{clean_transcript}' due to echo cooldown ({cooldown_elapsed:.2f}s < 1.2s).")
                                continue

                            if not self.call_ended:
                                self.is_bot_speaking = True
                                asyncio.create_task(self.process_llm_and_speak(clean_transcript))
        except Exception as e:
            if not self.call_ended:
                logger.error(f"[Exotel] Deepgram receive error: {e}")
        finally:
            self.deepgram_ws = None
            if not self.call_ended and self._deepgram_reconnect_count < 2:
                self._deepgram_reconnect_count += 1
                logger.warning(f"[Exotel] Deepgram connection dropped — reconnect attempt {self._deepgram_reconnect_count}/2")
                try:
                    await self.connect_to_deepgram()
                except Exception as e:
                    logger.error(f"[Exotel] Deepgram reconnect failed: {e}")
            elif not self.call_ended:
                logger.error("[Exotel] Deepgram reconnect limit reached — bot is deaf for remainder of call")

    async def _speak_text(self, text: str) -> float:
        """Convert text to speech and stream to Exotel (utility method). Returns audio duration in seconds.
        Automatically detects SSML markup (<speak> tags) and tells Polly to parse it.
        """
        duration_seconds = 0.0
        try:
            # Detect whether the text is SSML (wrapped in <speak> tags)
            is_ssml = text.strip().startswith("<speak>") and text.strip().endswith("</speak>")

            def fetch_polly():
                polly_params = {
                    "OutputFormat": "pcm",
                    "SampleRate": "8000",
                    "VoiceId": self.agent_voice,
                    "LanguageCode": self.agent_lang,
                    "Engine": "neural",
                }
                if is_ssml:
                    polly_params["Text"] = text
                    polly_params["TextType"] = "ssml"
                else:
                    # Wrap plain text in SSML with gentle pauses at punctuation
                    # This makes ALL LLM output sound more natural without changing prompts
                    ssml_text = text
                    # Add a micro-pause after "..." (the LLM's natural pause marker)
                    ssml_text = ssml_text.replace("...", '<break time="350ms"/>')
                    # Add a micro-pause after "।" (Devanagari full stop)
                    ssml_text = ssml_text.replace("।", '।<break time="250ms"/>')
                    polly_params["Text"] = f"<speak>{ssml_text}</speak>"
                    polly_params["TextType"] = "ssml"

                response = polly_client.synthesize_speech(**polly_params)
                stream = response.get("AudioStream")
                return stream.read() if stream else None

            pcm_data = await asyncio.to_thread(fetch_polly)
            
            if pcm_data:
                # 8000 samples/sec, 16-bit (2 bytes) = 16000 bytes/sec
                duration_seconds = len(pcm_data) / 16000.0
                
                pcm_chunks = chunk_pcm(pcm_data, chunk_size=1600)
                for pcm_b64 in pcm_chunks:
                    if self.call_ended or self.welcome_interrupted:
                        break
                    payload = {
                        "event": "media",
                        "streamSid": self.stream_sid,
                        "media": {
                            "payload": pcm_b64
                        }
                    }
                    logger.info(f"[Exotel] Sending outbound chunk {self.outbound_chunk_index} | timestamp: {self.outbound_timestamp_ms} ms")
                    await self.exotel_ws.send_json(payload)
                    self.outbound_chunk_index += 1
                    self.outbound_timestamp_ms += 100  # 100ms per 1600-byte chunk at 8kHz 16-bit PCM
                    # Pace the streaming to real-time (slightly less than 100ms to prevent underrun)
                    await asyncio.sleep(0.09)
        except Exception as e:
            logger.error(f"[Exotel] _speak_text error: {e}")
        return duration_seconds

    async def play_welcome_message(self, customer_name: str, bot_type: str):
        """Play the opening welcome message immediately without calling Groq first, reducing latency.
        Uses the two-beat sales psychology system for natural pauses and context-aware hooking.
        """
        if not hasattr(self, 'state_machine') or self.state_machine is None:
            self.state_machine = VoiceStateMachine(
                bot_type=bot_type, 
                customer_name=customer_name, 
                customer_category=getattr(self, 'customer_category', '')
            )

        def generate_welcome_message(b_type: str, c_name: str, crm_context: dict = None) -> str:
            ctx = crm_context or {}
            profession    = ctx.get("profession", "")
            business_name = ctx.get("business_name", "")
            city          = ctx.get("city", "Pune")
            gender        = ctx.get("gender", "M")

            # ── Salutation (works for all three) ──────────────────────────────────
            if business_name:
                greeting_line = f'Namaste, <break time="150ms"/> क्या यह {business_name} है?'
            else:
                _BUSINESS_SIGNALS = {
                    "pvt", "ltd", "llp", "inc", "corp", "limited",
                    "clinic", "hospital", "pharmacy", "medical", "dental", "lab",
                    "dr", "dr.", "doctor", "centre", "center",
                    "enterprise", "enterprises", "traders", "trading", "agency",
                    "store", "shop", "mart", "school", "college", "institute", "foundation", "trust",
                    "associates", "works", "care", "studio"
                }
                name_lower = c_name.lower() if c_name else ""
                tokens = name_lower.replace(".", " ").split()
                is_business = bool(set(tokens) & _BUSINESS_SIGNALS)
                if is_business:
                    greeting_line = f'Namaste, <break time="150ms"/> क्या यह {c_name} है?'
                else:
                    greeting_line = f'Namaste {c_name} ji? <break time="150ms"/>' if c_name else f'Namaste? <break time="150ms"/>'

            # Name Verification Check: Just return the greeting, don't include the hook.
            # We add a slight interrogative pause so they respond.
            return f'<speak>{greeting_line}</speak>'

        welcome_text = generate_welcome_message(bot_type, customer_name, getattr(self, "crm_context", {}))
                
        self.state_machine.state = CallState.VERIFY_NAME
        # Clean SSML tags for chat history
        clean_welcome = welcome_text
        clean_welcome = re.sub(r'<[^>]+>', '', clean_welcome)
        clean_welcome = re.sub(r'\s+', ' ', clean_welcome).strip()
        self.state_machine.chat_history.append({"role": "assistant", "content": clean_welcome})
        logger.info(f"[Exotel] Welcoming customer directly: {welcome_text}")
        
        self.is_bot_speaking = True
        try:
            await self._speak_text(welcome_text)
        finally:
            self.last_speak_finished_time = 0.0 if self.welcome_interrupted else asyncio.get_event_loop().time()
            self.is_bot_speaking = False
            
            if hasattr(self, 'barge_in_transcript') and self.barge_in_transcript and not self.call_ended:
                transcript_to_process = self.barge_in_transcript
                self.barge_in_transcript = ""
                asyncio.create_task(self.process_llm_and_speak(transcript_to_process))
                
            self.welcome_interrupted = False

    async def process_llm_and_speak(self, user_text: str):
        """Pass user text to Groq, then to Polly, then stream PCM back to Exotel.
        Detects [CALL_END] tag in LLM output and triggers hangup after final audio.
        """
        if self.call_ended:
            return

        self.is_bot_speaking = True
        
        if not hasattr(self, 'state_machine') or self.state_machine is None:
            self.state_machine = VoiceStateMachine(
                bot_type=getattr(self, 'bot_type', 'investment'), 
                customer_name=getattr(self, 'customer_name', ''), 
                customer_category=getattr(self, 'customer_category', '')
            )

        try:
            # 1. Update history with the user input
            self.state_machine.chat_history.append({"role": "user", "content": user_text})

            # Keep history manageable — system prompt + last 10 messages
            if len(self.state_machine.chat_history) > 12:
                self.state_machine.chat_history = (
                    [self.state_machine.chat_history[0]]  # system prompt
                    + self.state_machine.chat_history[-10:]  # recent context
                )

            instruction = self.state_machine.get_instruction_for_current_state(user_text=user_text)
            
            call_is_ending = False

            if self.state_machine.state == CallState.HANGUP and instruction.startswith("The call has gone on too long"):
                bot_text = "माफ़ कीजिएगा, यह call काफी लंबी हो गई है। मैं आपको बाद में कॉल करूँगी। [CALL_END]"
                self.state_machine.chat_history.append({"role": "assistant", "content": bot_text})
                logger.info(f"[Exotel] Bot (Max Turns): {bot_text}")
                call_is_ending = True
            else:
                messages = list(self.state_machine.chat_history)
                messages.append({"role": "system", "content": instruction})

                # 2. Get LLM Response from Groq (with 1 retry on transient errors)
                bot_text = None
                for attempt in range(2):
                    try:
                        response = await groq_client.chat.completions.create(
                            model=config.LLM_MODEL or "llama-3.3-70b-versatile",
                            messages=messages,
                            temperature=0.3,
                            max_tokens=150
                        )
                        bot_text = response.choices[0].message.content.strip()
                        break  # Success
                    except Exception as llm_err:
                        if attempt == 0:
                            logger.warning(f"[Exotel] LLM attempt 1 failed: {llm_err}. Retrying in 1s...")
                            await asyncio.sleep(1.0)
                        else:
                            raise  # Let outer except handle it

                self.state_machine.chat_history.append({
                    "role": "assistant",
                    "content": bot_text
                })
                logger.info(f"[Exotel] Bot: {bot_text}")

                # Post-process: let state machine parse tags and transition
                self.state_machine.post_process_response(bot_text)

                if "[CALL_END]" in bot_text or self.state_machine.state == CallState.HANGUP:
                    call_is_ending = True
                    self.state_machine.state = CallState.HANGUP
            
            speak_text = re.sub(r'\[(?:CALL[\s_]*END|END[\s_]*CALL)\]', '', bot_text, flags=re.IGNORECASE).strip()
            speak_text = speak_text.replace("[HANG_UP]", "").strip()
            speak_text = re.sub(r'\[APPOINTMENT:.*?\]', '', speak_text, flags=re.IGNORECASE).strip()
            speak_text = re.sub(r'\[LEAD:.*?\]', '', speak_text, flags=re.IGNORECASE).strip()

            duration_seconds = 0.0
            if speak_text:
                # Convert Text to Speech using AWS Polly (PCM output)
                duration_seconds = await self._speak_text(speak_text)

            # Since _speak_text now streams chunks in real-time, it already takes duration_seconds to run.
            # We add a small post-speech pause (e.g. 0.5s) for natural conversation flow.
            await asyncio.sleep(0.5)

            # 5. If [CALL_END] was detected or we moved to HANGUP, close sockets
            if call_is_ending:
                logger.info(f"[Exotel] Call Ending triggered — hanging up.")
                await asyncio.sleep(3.5)  # Buffer to allow Exotel to finish playing audio
                await self._hangup()

        except Exception as e:
            logger.error(f"[Exotel] Error in LLM/TTS pipeline: {e}")
            # Speak a fallback so the user doesn't hear dead silence
            try:
                await self._speak_text(
                    '<speak>माफ़ कीजिए, <break time="250ms"/> एक छोटी सी technical issue आ गई। '
                    '<break time="300ms"/> संजीव sir आपको जल्दी call करेंगे। '
                    '<break time="200ms"/> धन्यवाद!</speak>'
                )
                await asyncio.sleep(3.5)
                await self._hangup()
            except Exception:
                pass
        finally:
            self.last_speak_finished_time = 0.0 if self.welcome_interrupted else asyncio.get_event_loop().time()
            self.is_bot_speaking = False
            
            if hasattr(self, 'barge_in_transcript') and self.barge_in_transcript and not self.call_ended:
                transcript_to_process = self.barge_in_transcript
                self.barge_in_transcript = ""
                asyncio.create_task(self.process_llm_and_speak(transcript_to_process))

            self.welcome_interrupted = False

    async def _score_and_log_lead(self):
        """Extract conversation signals, log the scored lead to CRM, and save full transcript."""
        try:
            from server.lead_scoring import LeadData, BotType, LeadSource, score_lead
            from server.sheets_manager import sheets_manager, whatsapp_notifier

            # Build conversation transcript
            transcript_lines = []
            for m in self.state_machine.chat_history:
                if m['role'] in ('user', 'assistant'):
                    role = m['role']
                    content = m['content']
                    if role == 'assistant':
                        try:
                            parsed = json.loads(content)
                            content = parsed.get("response") or parsed.get("reply") or parsed.get("text") or content
                        except Exception:
                            pass
                    transcript_lines.append(f"{role}: {content}")
            transcript = "\n".join(transcript_lines)

            # Extract scheduled time from state machine
            scheduled_day = self.state_machine.scheduled_day
            scheduled_time = self.state_machine.scheduled_time

            # Determine call outcome
            if self.state_machine.scheduled_day and self.state_machine.scheduled_time:
                outcome = "callback_scheduled"
            elif self.state_machine.scheduled_day:
                outcome = "callback_agreed"
            elif self.state_machine.state == CallState.HANGUP:
                outcome = "not_interested"
            else:
                outcome = "incomplete"

            # Map bot_type string to BotType enum
            bot_type_map = {"investment": BotType.INVESTMENT, "insurance": BotType.INSURANCE}
            lead = LeadData(
                name=self.customer_name,
                phone=getattr(self, 'caller_phone', ''),
                conversation_summary=transcript[:500],
                source=LeadSource.OUTBOUND_CALL,
                bot_type=bot_type_map.get(self.bot_type, BotType.INVESTMENT),
                asked_for_callback=(self.state_machine.state == CallState.CONFIRM),
                ready_to_buy=(self.state_machine.state == CallState.CONFIRM),
                said_not_interested=(outcome == "not_interested"),
            )
            lead = score_lead(lead)

            # --- Save full transcript to disk ---
            try:
                call_log_dir = Path("data/call_logs")
                call_log_dir.mkdir(parents=True, exist_ok=True)
                call_log = {
                    "timestamp": datetime.now().isoformat(),
                    "phone": getattr(self, 'caller_phone', ''),
                    "customer_name": self.customer_name,
                    "customer_category": self.customer_category,
                    "bot_type": self.bot_type,
                    "outcome": outcome,
                    "scheduled_day": scheduled_day,
                    "scheduled_time": scheduled_time,
                    "lead_score": lead.score,
                    "lead_category": lead.category.value,
                    "full_transcript": transcript,
                }
                log_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{getattr(self, 'caller_phone', 'unknown')[-4:]}.json"
                log_path = call_log_dir / log_filename
                log_path.write_text(json.dumps(call_log, indent=2, ensure_ascii=False), encoding="utf-8")
                logger.info(f"[Exotel] Call transcript saved: {log_path}")
            except Exception as e:
                logger.error(f"[Exotel] Failed to save transcript to disk: {e}")

            # --- Log to CRM + WhatsApp ---
            if lead.category.value == "HOT":
                sheets_manager.log_hot_lead(lead, scheduled_day=scheduled_day, scheduled_time=scheduled_time)
                await whatsapp_notifier.notify_manager_hot_lead(lead, scheduled_day=scheduled_day, scheduled_time=scheduled_time)
                logger.info(f"[Exotel] 🔴 HOT LEAD logged: {lead.name or 'Unknown'} (Score: {lead.score})")
            elif lead.category.value == "WARM":
                sheets_manager.log_nurture_lead(lead)
                logger.info(f"[Exotel] 🟡 WARM LEAD logged: {lead.name or 'Unknown'} (Score: {lead.score})")
            else:
                logger.info(f"[Exotel] ⚪ COLD/DNC lead: Score {lead.score}")
        except Exception as e:
            logger.error(f"[Exotel] Error scoring/logging lead: {e}")

    async def _hangup(self):
        """Gracefully end the call by closing WebSocket connections."""
        if self.call_ended:
            return
        self.call_ended = True
        logger.info("[Exotel] Initiating graceful hangup...")

        # Score and log the lead before closing
        if self.state_machine:
            await self._score_and_log_lead()

        # Cancel silence watchdog
        if self.silence_watchdog_task and not self.silence_watchdog_task.done():
            self.silence_watchdog_task.cancel()

        # Close Deepgram connection
        if self.deepgram_ws:
            try:
                await self.deepgram_ws.close()
                logger.info("[Exotel] Deepgram connection closed.")
            except Exception:
                pass
            self.deepgram_ws = None

        # Close Exotel WebSocket — this tells Exotel to hang up the call
        try:
            await self.exotel_ws.close()
            logger.info("[Exotel] WebSocket closed — call ended.")
        except Exception:
            pass

    async def handle_exotel_message(self, message: str):
        """Handle incoming messages from Exotel WebSocket."""
        if self.call_ended:
            return

        data = json.loads(message)
        event = data.get("event")

        if event == "connected":
            logger.info("[Exotel] WebSocket connected.")

        elif event == "start":
            # Exotel uses snake_case stream_sid at root level, but check nested start and camelCase as fallback
            self.stream_sid = (
                data.get("stream_sid") or 
                data.get("streamSid") or 
                data.get("start", {}).get("stream_sid") or 
                data.get("start", {}).get("streamSid")
            )
            logger.info(f"[Exotel] Stream started. SID: {self.stream_sid} | Payload: {data}")
            
            # Dynamically set persona
            start_payload = data.get("start") or data or {}
            await self._determine_bot_persona(start_payload)
            
            await self.connect_to_deepgram()

            # Trigger welcome message directly in background (bypass Groq)
            customer_name = getattr(self, 'customer_name', '')
            bot_type = getattr(self, 'bot_type', 'investment')
            asyncio.create_task(self.play_welcome_message(customer_name, bot_type))

        elif event == "media":
            # Pass raw PCM audio to Deepgram (no decoding needed beyond base64)
            # We forward audio even when the bot is speaking to prevent Deepgram from timing out
            if self.deepgram_ws and not self.call_ended:
                payload = data.get("media", {}).get("payload")
                if payload:
                    try:
                        audio_bytes = base64.b64decode(payload)
                        await self.deepgram_ws.send(audio_bytes)
                    except websockets.exceptions.ConnectionClosed as e:
                        logger.warning(f"[Exotel] Deepgram connection closed while sending media: {e}")
                        self.deepgram_ws = None
                    except Exception as e:
                        logger.error(f"[Exotel] Error sending media to Deepgram: {e}")

        elif event == "stop":
            logger.info("[Exotel] Stream stopped by Exotel.")
            await self._hangup()
        else:
            logger.warning(f"[Exotel] Unhandled WebSocket event: {event} | Payload: {data}")
