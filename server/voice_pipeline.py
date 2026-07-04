import re
import json
import base64
import asyncio
import logging
from pathlib import Path
from datetime import datetime
import websockets
import boto3
from server.config import config
from server.audio_utils import chunk_pcm
from server.voice_state_machine import CallState, VoiceStateMachine, classify_bot_type, score_lead_with_llm

logger = logging.getLogger(__name__)

import csv

def _append_to_qa_csv(call_log: dict):
    """Helper function to log every call to the master QA CSV."""
    qa_file = Path("data/qa_call_logs_master.csv")
    qa_file.parent.mkdir(parents=True, exist_ok=True)
    file_exists = qa_file.exists()
    
    headers = [
        "timestamp", "phone", "customer_name", "customer_category", "bot_type",
        "outcome", "scheduled_day", "scheduled_time", "lead_score", "lead_category",
        "llm_interest", "llm_objection", "llm_summary", "full_transcript"
    ]
    
    try:
        with open(qa_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow(call_log)
    except Exception as e:
        logger.error(f"Failed to append to QA CSV: {e}")


# ── Spoken-text guards ─────────────────────────────────────────────────────
# The prompt asks the LLM for short, human replies, but a 70B model at temp 0.5
# regularly overshoots. These guards enforce brevity + kill AI-chatbot filler in
# Python — the ONLY reliable place — right before the text reaches Polly TTS.

# Sentence boundary: Hindi danda plus the Latin terminators.
_SENTENCE_END = re.compile(r'(?<=[।.!?])\s+')

# English "AI assistant" filler that instantly makes the bot sound like a chatbot,
# not a person on the phone. Stripped wholesale (Hindi equivalents are handled in
# the prompt, which is the better lever for Devanagari).
_AI_ISH_PATTERNS = [
    (re.compile(r"\bI(?:'m| am)?\s+(?:completely |totally |fully |absolutely )?(?:understand|here to help)\b[^.।!?]*[.।!?]?\s*", re.I), ""),
    (re.compile(r"\bI(?:'| a|'?d)?\s*(?:would |'d )?be (?:more than )?happy to\b\s*", re.I), ""),
    (re.compile(r"\b(?:please )?feel free to\b\s*", re.I), ""),
    (re.compile(r"\brest assured\b[,.]?\s*", re.I), ""),
    (re.compile(r"\bas (?:I|we) (?:mentioned|discussed) (?:earlier|before)?\b[,.]?\s*", re.I), ""),
    (re.compile(r"\bthat(?:'s| is) a (?:great|good|wonderful|excellent) question\b[,.!]?\s*", re.I), ""),
    (re.compile(r"\bI hope (?:this|that) helps\b[,.!]?\s*", re.I), ""),
    (re.compile(r"\bat your convenience\b", re.I), "jab aapko time ho"),
]


def _strip_ai_ish(text: str) -> str:
    """Remove canned assistant filler that makes the voice sound robotic."""
    for pat, repl in _AI_ISH_PATTERNS:
        text = pat.sub(repl, text)
    return re.sub(r"\s{2,}", " ", text).strip()


def _shorten_for_voice(text: str, max_words: int = 26) -> str:
    """Hard brevity guard for TTS output.

    - Keeps whole sentences up to `max_words`, so we never cut mid-sentence.
    - Drops a trailing fragment with no terminator (a telltale LLM token cutoff)
      as long as at least one complete sentence remains.
    - Falls back to a clean word-boundary cut for a single run-on sentence.
    """
    text = text.strip()
    if not text or len(text.split()) <= max_words:
        return text

    sentences = [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]
    if len(sentences) > 1 and sentences[-1][-1:] not in "।.!?":
        sentences = sentences[:-1]

    kept, count = [], 0
    for s in sentences:
        w = len(s.split())
        if kept and count + w > max_words:
            break
        kept.append(s)
        count += w
    result = " ".join(kept).strip()

    # A single sentence still over the cap → clean cut at the word boundary.
    if not result or len(result.split()) > max_words:
        result = " ".join((result or text).split()[:max_words]).rstrip(" ,;:—-")
        if result and result[-1] not in "।.!?":
            result += "।"
    return result


def _clean_speak_text(bot_text: str) -> str:
    """Strip control tags, remove AI-ish filler, and hard-cap length for TTS."""
    t = re.sub(r"\[(?:CALL[\s_]*END|END[\s_]*CALL)\]", "", bot_text, flags=re.IGNORECASE)
    t = t.replace("[HANG_UP]", "").replace("[RECOVERY]", "")
    t = re.sub(r"\[APPOINTMENT:.*?\]", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\[LEAD:.*?\]", "", t, flags=re.IGNORECASE)
    t = _strip_ai_ish(t.strip())
    return _shorten_for_voice(t).strip()


# Initialize AWS Polly Client
polly_client = boto3.client(
    "polly",
    aws_access_key_id=config.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
    region_name=config.AWS_REGION
)

# LLM completion (OpenRouter primary → Groq fallback) lives in the shared llm_client
# module, so the pipeline and the post-call scorer share ONE provider/fallback policy.
from server.llm_client import complete as _llm_complete


class ExotelVoiceConnectionManager:
    """
    Handles bidirectional audio streaming between Exotel and the AI pipeline.
    Exotel streams 16-bit Linear PCM, so we stream raw PCM straight through
    (no mu-law conversion needed).

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
        # On Exotel OUTBOUND calls the customer is the `from` leg — `to` is our own
        # ExoPhone/DID. Prioritise `from` so lead lookup + CRM logging use the real
        # number.
        phone = start_data.get("from") or start_data.get("to") or ""
        custom_params = start_data.get("customParameters", {}) or start_data.get("custom_parameters", {}) or {}

        # Exotel delivers a CustomField JSON blob as a single HTML-escaped key, e.g.
        #   {'{&quot;bot_type&quot;: &quot;insurance&quot;, &quot;customer_name&quot;: &quot;X&quot;}': ''}
        # Unwrap it back into a real dict so bot_type/name/category are honoured.
        if isinstance(custom_params, dict) and len(custom_params) == 1:
            only_key = next(iter(custom_params))
            if isinstance(only_key, str) and only_key.strip().startswith("{"):
                import html
                try:
                    custom_params = json.loads(html.unescape(only_key))
                except Exception:
                    pass

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
        lead_files = [
            Path("data/leads/hni_leads_pune.csv"),
            Path("data/leads/unified_compliant_leads.csv")
        ]
        for leads_file in lead_files:
            if leads_file.exists() and clean_phone and not customer_name:
                try:
                    import csv
                    with open(leads_file, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            row_phone = normalize_to_10_digits(row.get("phone", ""))
                            if row_phone and row_phone == clean_phone:
                                customer_name = row.get("name", "").strip()
                                customer_category = (row.get("category", "") or row.get("profession", "")).strip()
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
                    # Let Polly's neural voice handle sentence pauses itself. Injecting a
                    # <break> after every "।" plus ellipsis breaks made speech choppy —
                    # callers hear that as "voice breaking". Strip ellipses; no extra breaks.
                    ssml_text = text.replace("...", " ").replace("…", " ")
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
                # NOTE: no logging inside this loop — per-chunk log I/O (~10/sec) stalls
                # the send loop and causes choppy/breaking audio. Summary log after instead.
                interrupted = False
                for pcm_b64 in pcm_chunks:
                    if self.call_ended:
                        break
                    if self.welcome_interrupted:
                        interrupted = True
                        break
                    # Exotel AgentStream requires snake_case "stream_sid" (NOT camelCase
                    # "streamSid"). With the wrong key Exotel silently drops the frame
                    # and the caller hears dead air.
                    payload = {
                        "event": "media",
                        "stream_sid": self.stream_sid,
                        "media": {
                            "payload": pcm_b64
                        }
                    }
                    await self.exotel_ws.send_json(payload)
                    self.outbound_chunk_index += 1
                    self.outbound_timestamp_ms += 100  # 100ms per 1600-byte chunk at 8kHz 16-bit PCM
                    # Pace the streaming to real-time (slightly less than 100ms to prevent underrun)
                    await asyncio.sleep(0.09)

                # Barge-in: the caller started speaking while Riya was talking. Stopping our
                # send loop isn't enough — Exotel still has ~1s of audio buffered and will keep
                # playing it (Riya talks over the caller). Send a `clear` event to flush that
                # buffer so she stops instantly, like a real person would.
                if interrupted:
                    try:
                        await self.exotel_ws.send_json(
                            {"event": "clear", "stream_sid": self.stream_sid}
                        )
                        logger.info("[Exotel] Barge-in — sent clear to flush playout buffer.")
                    except Exception as e:
                        logger.debug(f"[Exotel] clear on barge-in failed: {e}")
                logger.debug(f"[Exotel] Streamed {len(pcm_chunks)} chunks (~{duration_seconds:.1f}s audio)")
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
            # CRISP opener: greet + who + why + ask for a moment — all in one, no waiting
            # for a "Namaste" back-and-forth. Plain words, ~20 words max.
            _BUSINESS_SIGNALS = {
                "pvt", "ltd", "llp", "clinic", "hospital", "pharmacy", "medical", "dental",
                "lab", "dr", "doctor", "centre", "center", "enterprises", "traders", "agency",
                "store", "shop", "school", "college", "institute", "trust", "associates", "studio", "care",
            }
            tokens = (c_name or "").lower().replace(".", " ").split()
            is_business = bool(set(tokens) & _BUSINESS_SIGNALS)
            name_part = f"{c_name} जी, " if (c_name and not is_business) else ""

            if b_type == "insurance":
                who, why = "मैं Aarav, Kalpvruksh Finserv Pune से", "आपके health cover पर एक ज़रूरी बात बतानी थी"
            elif b_type == "recruitment":
                who, why = "मैं Riya, Kalpvruksh Finserv Pune से", "एक extra income का आसान मौका बताना था"
            else:  # investment
                who, why = "मैं Riya, Kalpvruksh Finserv Pune से", "आपके पैसे बढ़ाने का एक आसान तरीका बताना था"

            opener = f"नमस्ते {name_part}{who}। {why} — दो मिनट हैं आपके पास?"
            return f"<speak>{opener}</speak>"

        welcome_text = generate_welcome_message(bot_type, customer_name, getattr(self, "crm_context", {}))
                
        self.state_machine.state = CallState.OPENING
        # Clean SSML tags for chat history
        clean_welcome = welcome_text
        clean_welcome = re.sub(r'<[^>]+>', '', clean_welcome)
        clean_welcome = re.sub(r'\s+', ' ', clean_welcome).strip()
        self.state_machine.chat_history.append({"role": "assistant", "content": clean_welcome})
        self.state_machine.full_transcript.append({"role": "assistant", "content": clean_welcome})
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
            self.state_machine.full_transcript.append({"role": "user", "content": user_text})

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
                logger.info(f"[Exotel] Bot (Max Turns): {bot_text}")
                call_is_ending = True
            else:
                messages = list(self.state_machine.chat_history)
                messages.append({"role": "system", "content": instruction})

                # 2. Get LLM response — auto-falls back to a higher-limit model on
                #    rate-limit/error so a quota hit degrades quality, not drops the call.
                response = await _llm_complete(messages, temperature=0.5, max_tokens=100)
                bot_text = response.choices[0].message.content.strip()
                logger.info(f"[Exotel] Bot: {bot_text}")

                # Post-process: let state machine parse tags and transition
                self.state_machine.post_process_response(bot_text)

                if "[CALL_END]" in bot_text or self.state_machine.state == CallState.HANGUP:
                    call_is_ending = True
                    self.state_machine.state = CallState.HANGUP

            # Clean tags + enforce brevity/anti-AI-filler BEFORE TTS.
            speak_text = _clean_speak_text(bot_text)
            # Record the turn: history gets the SHORT spoken line so the model mirrors
            # this brevity next turn; the transcript keeps the raw text for scoring.
            self.state_machine.chat_history.append({"role": "assistant", "content": speak_text or bot_text})
            self.state_machine.full_transcript.append({"role": "assistant", "content": bot_text})

            duration_seconds = 0.0
            if speak_text:
                # Convert Text to Speech using AWS Polly (PCM output)
                duration_seconds = await self._speak_text(speak_text)

            # _speak_text already streams in real-time (takes duration_seconds). A short
            # post-speech pause keeps the flow natural without adding perceptible lag.
            await asyncio.sleep(0.2)

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
        """LLM-based lead scoring + CRM logging (Exotel)."""
        try:
            from server.lead_scoring import LeadData, BotType, LeadSource, LeadCategory
            from server.sheets_manager import sheets_manager, whatsapp_notifier

            # Build transcript
            transcript_lines = []
            for m in self.state_machine.full_transcript:
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

            # Determine outcome
            if scheduled_day and scheduled_time:
                outcome = "callback_scheduled"
            elif scheduled_day:
                outcome = "callback_agreed"
            else:
                outcome = "completed"

            # LLM-based scoring (happens AFTER call, no customer-facing latency)
            llm_score = await score_lead_with_llm(transcript, self.bot_type)
            logger.info(f"[Exotel] LLM Score: {llm_score}")

            # Build LeadData with LLM results
            category_map = {"HOT": LeadCategory.HOT, "WARM": LeadCategory.WARM, "COLD": LeadCategory.COLD, "DNC": LeadCategory.DNC}
            bot_type_map = {"investment": BotType.INVESTMENT, "insurance": BotType.INSURANCE, "recruitment": BotType.RECRUITMENT}
            lead = LeadData(
                name=self.customer_name,
                phone=getattr(self, 'caller_phone', ''),
                conversation_summary=llm_score.get("summary", transcript[:500]),
                source=LeadSource.OUTBOUND_CALL,
                bot_type=bot_type_map.get(self.bot_type, BotType.INVESTMENT),
                score=llm_score["score"],
                category=category_map.get(llm_score["category"], LeadCategory.COLD),
            )

            # Save full transcript to disk
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
                    "llm_interest": llm_score.get("interest", ""),
                    "llm_objection": llm_score.get("objection", ""),
                    "llm_summary": llm_score.get("summary", ""),
                    "full_transcript": transcript,
                }
                log_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{getattr(self, 'caller_phone', 'unknown')[-4:]}.json"
                log_path = call_log_dir / log_filename
                log_path.write_text(json.dumps(call_log, indent=2, ensure_ascii=False), encoding="utf-8")
                logger.info(f"[Exotel] Call transcript saved: {log_path}")
                _append_to_qa_csv(call_log)
                logger.info(f"[Exotel] Appended to QA CSV log")
            except Exception as e:
                logger.error(f"[Exotel] Failed to save transcript to disk: {e}")

            # Log to CRM + WhatsApp
            if lead.category == LeadCategory.DNC:
                # Caller asked not to be contacted — add to DND so we never call again
                from server.campaign.trai_compliance import add_to_dnd
                add_to_dnd(getattr(self, 'caller_phone', ''), reason="requested_no_contact")
                logger.info(f"[Exotel] 🚫 DNC — added {getattr(self, 'caller_phone', '')} to DND list")
            elif lead.category == LeadCategory.HOT:
                sheets_manager.log_hot_lead(lead, scheduled_day=scheduled_day, scheduled_time=scheduled_time)
                await whatsapp_notifier.notify_manager_hot_lead(lead, scheduled_day=scheduled_day, scheduled_time=scheduled_time)
                logger.info(f"[Exotel] 🔴 HOT LEAD logged: {lead.name or 'Unknown'} (Score: {lead.score})")
            elif lead.category == LeadCategory.WARM:
                sheets_manager.log_nurture_lead(lead)
                logger.info(f"[Exotel] 🟡 WARM LEAD logged: {lead.name or 'Unknown'} (Score: {lead.score})")
            else:
                logger.info(f"[Exotel] ⚪ {lead.category.value} lead: Score {lead.score}")
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
