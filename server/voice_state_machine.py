"""
Kalpvruksh Finserv — Voice Call State Machine

Philosophy: Python is the navigator. The LLM is the speaker.
- State transitions are deterministic Python — never delegated to the LLM.
- Per-turn situational instructions tell the LLM exactly what situation it is in
  and what its goal is for this turn, without bloating the system prompt.
- classify_permission() and extract_datetime() are lightweight async classifiers
  that pre-process user input before the main LLM call.
"""

import re
import json
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from enum import Enum
from groq import AsyncGroq
from server.config import config

logger = logging.getLogger(__name__)

groq_client = AsyncGroq(api_key=config.GROQ_API_KEY)

CLASSIFIER_MODEL  = "llama-3.3-70b-versatile"
CONVERSATION_MODEL = config.LLM_MODEL or "llama-3.3-70b-versatile"


# ── State Definitions ──────────────────────────────────────────────

class CallState(Enum):
    VERIFY_NAME      = "VERIFY_NAME"        # Verifying name; waiting for yes or name correction
    LANGUAGE_CHECK   = "LANGUAGE_CHECK"     # Asking language preference: Hindi or English
    CHECK_PERMISSION = "CHECK_PERMISSION"   # Hook delivered; waiting for permission to continue
    QUALIFY          = "QUALIFY"            # Discovery and curiosity building
    SCHEDULE         = "SCHEDULE"           # Collecting day + time for Sanjeev sir's callback
    CONFIRM          = "CONFIRM"            # Both day + time received; confirming appointment
    HANGUP           = "HANGUP"             # Final state; call ending

ALLOWED_TRANSITIONS = {
    "VERIFY_NAME":      {"LANGUAGE_CHECK", "HANGUP"},
    "LANGUAGE_CHECK":   {"CHECK_PERMISSION", "HANGUP"},
    "CHECK_PERMISSION": {"QUALIFY", "HANGUP"},
    "QUALIFY":          {"QUALIFY", "SCHEDULE", "HANGUP"},
    "SCHEDULE":         {"SCHEDULE", "CONFIRM", "HANGUP"},
    "CONFIRM":          {"CONFIRM", "HANGUP"},
    "HANGUP":           {"HANGUP"},
}

def apply_transition(current: str, requested: str) -> str:
    """Guard rail: only allows whitelisted transitions. Logs and blocks invalid ones."""
    allowed = ALLOWED_TRANSITIONS.get(current, set())
    if requested in allowed:
        return requested
    logger.warning(f"Blocked invalid transition: {current} → {requested}. Staying in {current}.")
    return current


# ── Date / Time Normalizers ────────────────────────────────────────

def normalize_scheduled_date(date_str: str) -> str | None:
    """Convert natural language dates ('kal', 'tomorrow', 'Monday') to YYYY-MM-DD in IST."""
    if not date_str or str(date_str).strip().lower() in ("null", "none", ""):
        return None
    s = str(date_str).strip().lower()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s
    ist = ZoneInfo("Asia/Kolkata")
    now = datetime.now(ist)
    if "today" in s or "aaj" in s:
        return now.strftime("%Y-%m-%d")
    if "tomorrow" in s or "kal" in s:
        return (now + timedelta(days=1)).strftime("%Y-%m-%d")
    if "day after" in s or "parso" in s or "parson" in s:
        return (now + timedelta(days=2)).strftime("%Y-%m-%d")
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for i, day in enumerate(days):
        if day in s:
            ahead = i - now.weekday()
            if ahead <= 0 or "next" in s:
                ahead += 7
            return (now + timedelta(days=ahead)).strftime("%Y-%m-%d")
    return None

def normalize_scheduled_time(time_str: str) -> str | None:
    """Convert natural language times ('shaam 5 baje', '5 PM') to HH:MM (24-hour)."""
    if not time_str or str(time_str).strip().lower() in ("null", "none", ""):
        return None
    s = str(time_str).strip().upper()
    if re.match(r"^\d{2}:\d{2}$", s):
        return s
    try:
        if "PM" in s or "AM" in s:
            return datetime.strptime(s.replace(" ", ""), "%I:%M%p").strftime("%H:%M")
    except ValueError:
        try:
            return datetime.strptime(s.replace(" ", ""), "%I%p").strftime("%H:%M")
        except ValueError:
            pass
    return s  # Return as-is when parsing fails — still usable for display


# ── Async LLM Classifiers ──────────────────────────────────────────

async def classify_permission(user_text: str) -> str:
    """
    Classify user intent as YES / NO / MAYBE.
    Used for: permission check (CHECK_PERMISSION state) and
    scheduling agreement (QUALIFY → SCHEDULE transition).

    YES   = agreement, interest, curiosity, questions (curiosity = soft yes)
    NO    = clear refusal, go away, remove number, not interested
    MAYBE = hesitant, busy but not refusing, asking for more info, pushback/skepticism
    """
    prompt = (
        "You are classifying intent from an Indian voice call (Hindi/Hinglish/English).\n"
        "Output exactly one word: YES, NO, or MAYBE.\n\n"
        "YES  → agreement, interest, curiosity, willingness, asking questions out of genuine interest\n"
        "       Examples: haan batao, achha, interesting, tell me more, kya offer hai, ji boliye\n\n"
        "NO   → HARD refusal, anger, or explicit rejection ONLY:\n"
        "       Examples: nahi chahiye, not interested, number remove karo, call mat karna,\n"
        "       timepass mat karo, scam hai, fraud, phone rakh do, bakwaas band karo\n\n"
        "MAYBE → EVERYTHING ELSE: hesitation, skepticism, defensive questions, pushback,\n"
        "        busy, confused, asking why you called, challenging your question,\n"
        "        asking who you are, requesting credentials\n"
        "        Examples: kya chahiye, kyun pooch rahe ho, kaun bol raha hai,\n"
        "        abhi busy hoon, baad mein baat karo, socha nahi, dekhenge,\n"
        "        why are you asking, what is this about, mujhe kaise pata\n\n"
        "IMPORTANT: If in doubt between NO and MAYBE, always choose MAYBE.\n"
        "Only use NO for absolutely clear, hostile rejections.\n\n"
        f'User said: "{user_text}"\n\nClassification:'
    )
    try:
        res = await groq_client.chat.completions.create(
            model=CLASSIFIER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5,
        )
        result = re.sub(r"[^A-Z]", "", res.choices[0].message.content.strip().upper())
        return result if result in ("YES", "NO", "MAYBE") else "MAYBE"
    except Exception as e:
        logger.error(f"classify_permission error: {e}")
        return "MAYBE"  # Safe default — keeps call alive


async def classify_language(user_text: str) -> str:
    """
    Classify whether the user prefers Hindi/Hinglish or English.
    Returns: HINDI or ENGLISH.
    """
    prompt = (
        "The user was asked: 'Hindi mein baat karein ya English mein?'\n"
        "Based on their response, classify their language preference.\n"
        "Output exactly one word: HINDI or ENGLISH.\n\n"
        "HINDI   → They said hindi, haan, ji, theek hai, chalega, Hindi mein, हां, हिंदी, or any affirmative/unclear response\n"
        "ENGLISH → They explicitly asked for English: english, english please, english mein, angrezi\n\n"
        "Default: HINDI (if unclear, assume Hindi since this is an Indian market)\n\n"
        f'User said: "{user_text}"\n\nClassification:'
    )
    try:
        res = await groq_client.chat.completions.create(
            model=CLASSIFIER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5,
        )
        result = re.sub(r"[^A-Z]", "", res.choices[0].message.content.strip().upper())
        return result if result in ("HINDI", "ENGLISH") else "HINDI"
    except Exception as e:
        logger.error(f"classify_language error: {e}")
        return "HINDI"  # Safe default for Indian market


async def extract_datetime(user_text: str) -> dict:
    """
    Extract appointment day and time from user speech.
    Returns: {"day": "YYYY-MM-DD or None", "time": "HH:MM or None"}
    """
    prompt = (
        "Extract appointment scheduling info from this voice call response.\n"
        "Respond ONLY with valid JSON — no markdown, no explanation.\n\n"
        '{"day": "<today|tomorrow|day after tomorrow|Monday|...> or null", '
        '"time": "<5 PM|10:30 AM|2 PM|...> or null"}\n\n'
        "Extract only what is explicitly mentioned. Null if not mentioned.\n\n"
        f'User said: "{user_text}"\n\nJSON:'
    )
    try:
        res = await groq_client.chat.completions.create(
            model=CLASSIFIER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=50,
        )
        raw = re.sub(r"```json|```", "", res.choices[0].message.content.strip()).strip()
        parsed = json.loads(raw)
        return {
            "day":  normalize_scheduled_date(parsed.get("day")),
            "time": normalize_scheduled_time(parsed.get("time")),
        }
    except Exception as e:
        logger.error(f"extract_datetime error: {e}")
        return {"day": None, "time": None}


async def classify_bot_type(category: str) -> str:
    """
    Route a lead to the correct bot based on their profession/category.
    Uses LLM semantics — no hardcoded keyword dictionaries.
    Defaults to 'investment' when uncertain.
    """
    if not category or not category.strip():
        return "investment"
    prompt = (
        "Route this lead to the correct sales bot. Output ONE word only.\n\n"
        "INVESTMENT  → business owners, salaried professionals, HNIs, anyone needing wealth/financial planning\n"
        "INSURANCE   → healthcare workers, doctors, dentists, clinics, hospitals, medical field\n"
        "RECRUITMENT → existing financial advisors, insurance agents, MFDs, CAs, wealth managers\n"
        "Default: INVESTMENT when unsure.\n\n"
        f'Lead category: "{category}"\n\nAnswer:'
    )
    try:
        res = await groq_client.chat.completions.create(
            model=CLASSIFIER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        result = re.sub(r"[^A-Z]", "", res.choices[0].message.content.strip().upper())
        return {"INVESTMENT": "investment", "INSURANCE": "insurance", "RECRUITMENT": "recruitment"}.get(result, "investment")
    except Exception as e:
        logger.error(f"classify_bot_type error: {e}")
        return "investment"


class VoiceStateMachine:
    """
    One instance per call. Owns state, chat history, and scheduled appointment.

    v3 — Zero-classifier design:
    - Language detection: simple keyword match (no LLM needed)
    - Intent classification: baked into the instruction — the main LLM decides
    - State transitions: driven by turn count + LLM output tags ([CALL_END], [APPOINTMENT:...])
    - Result: ONE Groq call per turn instead of TWO → ~2s latency, not ~5s
    """

    def __init__(self, bot_type: str, customer_name: str = "", customer_category: str = ""):
        self.state             = CallState.CHECK_PERMISSION
        self.bot_type          = bot_type
        self.customer_name     = customer_name or ""
        self.customer_category = customer_category or ""
        self.language_preference = "hinglish"  # Updated by _detect_language
        self.scheduled_day     = None
        self.scheduled_time    = None
        self.qualify_turns     = 0   # Turns spent in QUALIFY — gates appointment offer
        self.total_turns       = 0   # Hard limit guard
        self.chat_history      = []
        self._initialize_persona()

    def _initialize_persona(self):
        """
        Load the correct prompt file and inject customer context.
        The system prompt goes in once at init; per-turn context is
        injected separately via get_instruction_for_current_state().
        """
        from pathlib import Path

        prompt_map = {
            "insurance":   "prompts/insurance_bot_prompt.txt",
            "recruitment": "prompts/advisor_recruitment_bot_prompt.txt",
            "investment":  "prompts/investment_bot_prompt.txt",
        }
        prompt_path = Path(prompt_map.get(self.bot_type, "prompts/investment_bot_prompt.txt"))
        try:
            identity_block = (
                prompt_path.read_text(encoding="utf-8")
                if prompt_path.exists()
                else f"You are a warm, helpful {self.bot_type} advisor at Kalpvruksh Finserv, Pune."
            )
        except Exception as e:
            logger.error(f"Failed to read prompt file '{prompt_path}': {e}")
            identity_block = f"You are a warm, helpful {self.bot_type} advisor at Kalpvruksh Finserv, Pune."

        # Name block — let the LLM use judgment; don't force awkward name usage
        if self.customer_name.strip():
            name_block = (
                f'CUSTOMER NAME FROM DATABASE: "{self.customer_name}"\n'
                'If it looks like a category, placeholder, or business term, use "आप" respectfully.'
            )
        else:
            name_block = 'Customer name unknown. Use "आप" respectfully throughout. Never guess or invent a name.'

        if self.customer_category.strip():
            name_block += (
                f'\nCUSTOMER CONTEXT: Category is "{self.customer_category}". '
                "Weave this into the conversation naturally — do NOT say the category word verbatim."
            )

        persona = f"""{identity_block}

TTS SCRIPT RULES (Critical for Polly pronunciation):
- Hindi/Hinglish words → Devanagari: मैं, आप, अच्छा, कल, ठीक है, बिल्कुल
- Financial/English terms → Latin: SIP, mutual funds, savings, insurance, consultation, Kalpvruksh Finserv
- Founder name in Devanagari: संजीव सुराना — never "Sandeep Khurana" or any other name
- NEVER transliterate Hindi into Latin script (never write "main", "aap", "achha")

OUTPUT RULES:
- Max 2 sentences, 30 words per turn. One question only. Then stop.
- NEVER repeat the customer's name during the conversation. Address them by name only at the very start or end. Repeatedly saying their name sounds robotic and irritating.
- Sound like a smart, caring friend — never a telecaller reading a script.
- To end call: append [CALL_END] to your final farewell sentence and nowhere else.
- To confirm appointment: output [APPOINTMENT: day=<YYYY-MM-DD>, time=<HH:MM>, name=<name or unknown>]
  on its own line, then the farewell + [CALL_END].

{name_block}"""

        self.chat_history.append({"role": "system", "content": persona})

    # ── Helpers ──────────────────────────────────────────────────────

    def _lang_note(self) -> str:
        """Return a brief language reminder for injection into every turn instruction."""
        if self.language_preference == "english":
            return "(Speak in natural English only — no Hindi.)"
        return "(Speak in Hinglish — Hindi words in Devanagari, English terms in Latin.)"

    def _detect_language(self, user_text: str):
        """Simple keyword-based language detection — zero latency, no LLM needed."""
        english_signals = {"english", "eng", "angrezi", "inglish", "in english"}
        text_lower = user_text.lower().strip()
        if any(kw in text_lower for kw in english_signals):
            self.language_preference = "english"
        else:
            self.language_preference = "hinglish"
        logger.info(f"Language detected from keywords: {self.language_preference}")

    def post_process_response(self, bot_text: str):
        """
        Called AFTER the main LLM responds. Handles state transitions
        based on tags in the LLM output ([CALL_END], [APPOINTMENT:...]).
        """
        # Parse [APPOINTMENT:...] tag if present
        appointment_match = re.search(
            r'\[APPOINTMENT:\s*day=([^,\]]+),\s*time=([^,\]]+)',
            bot_text, re.IGNORECASE
        )
        if appointment_match:
            raw_day = appointment_match.group(1).strip()
            raw_time = appointment_match.group(2).strip()
            self.scheduled_day = raw_day
            self.scheduled_time = normalize_scheduled_time(raw_time) if raw_time else raw_time
            if self.state not in (CallState.CONFIRM, CallState.HANGUP):
                self.state = CallState.CONFIRM
            logger.info(f"Appointment parsed: day={self.scheduled_day}, time={self.scheduled_time}")

        # [CALL_END] → force HANGUP
        if "[CALL_END]" in bot_text:
            self.state = CallState.HANGUP

    # ── Core Navigator ──────────────────────────────────────────────

    def get_instruction_for_current_state(self, user_text: str) -> str:
        """
        The core navigator. Called once per turn, before the main LLM call.

        v3 philosophy: Give the LLM ALL conditional paths in ONE instruction.
        The LLM reads the user's message and picks the right path based on
        full conversation context. No separate classifier needed.
        """
        self.total_turns += 1
        lang = self._lang_note()

        # Hard turn limit — cost control and troll protection
        if self.total_turns >= 15:
            self.state = CallState.HANGUP
            return f"The call has gone on too long. Thank them genuinely for their time and say goodbye warmly. {lang} Output your [LEAD:...] tag and append [CALL_END]."

        # ── VERIFY_NAME ──────────────────────────────────────────────
        if self.state == CallState.VERIFY_NAME:
            self.state = CallState.LANGUAGE_CHECK

            if self.bot_type == "insurance":
                bot_name = "Aarav"
                gender_form = "bol raha hoon"
            elif self.bot_type == "recruitment":
                bot_name = "Riya"
                gender_form = "bol rahi hoon"
            else:
                bot_name = "Riya"
                gender_form = "bol rahi hoon"

            return (
                "The user just responded to your 'Namaste'. "
                f"Introduce yourself warmly: 'Main Kalpvruksh Finserv se {bot_name} {gender_form}.' "
                "Then ask their language preference naturally: 'क्या आप Hindi में बात करना prefer करेंगे या English में?' "
                "If they corrected their name, acknowledge it warmly first. "
                "Keep it brief — just introduce + ask language. No pitch yet."
            )

        # ── LANGUAGE_CHECK ───────────────────────────────────────────
        if self.state == CallState.LANGUAGE_CHECK:
            # Detect language from user text — instant keyword match
            self._detect_language(user_text)
            self.state = CallState.CHECK_PERMISSION
            lang = self._lang_note()  # Refresh after detection

            if self.language_preference == "english":
                lang_instruction = (
                    "LANGUAGE MODE: The customer chose ENGLISH. "
                    "Speak in clean, natural English. No Hindi at all. Warm and conversational."
                )
            else:
                lang_instruction = (
                    "LANGUAGE MODE: The customer chose HINDI. "
                    "Speak in natural Hinglish — Hindi words in Devanagari, English terms in Latin."
                )

            # Reference the prompt's opening stage
            if self.bot_type == "insurance":
                hook_instruction = (
                    "Now follow your OPENING & DISCOVERY stage (Stage 1 in your prompt). "
                    "Ask one engaging question about their healthcare protection. "
                    "Be consultative, not salesy. Make them think."
                )
            elif self.bot_type == "recruitment":
                hook_instruction = (
                    "Now follow your STAGE 1 — BUSINESS HOOK from your prompt. "
                    "Create curiosity about the business partnership opportunity. "
                    "Generate a fresh, natural opening — don't recite the prompt."
                )
            else:  # investment
                hook_instruction = (
                    "Now follow your S1 — OPENING from your prompt. "
                    "Make them think about whether their savings strategy is aligned with rising future costs. "
                    "Generate a fresh question — never the same phrasing twice. "
                    "NEVER ask about specific amounts, income, salary, or portfolio value."
                )

            return f"{lang_instruction}\n\n{hook_instruction}"

        # ── CHECK_PERMISSION ─────────────────────────────────────────
        # One turn only — the LLM reads the user's reaction and decides how to proceed
        if self.state == CallState.CHECK_PERMISSION:
            self.state = CallState.QUALIFY  # Optimistic advance

            if self.bot_type == "insurance":
                return (
                    f"The user just responded to your opening hook. Read their response carefully. {lang} "
                    "Do NOT use their name in this turn.\n\n"
                    "• If they engaged, asked questions, or showed curiosity → follow your DISCOVERY stage. "
                    "Ask one open question about their current health coverage.\n"
                    "• If they seem confused, hesitant, or said 'not clear' → acknowledge warmly, DON'T repeat the same question. "
                    "Try a completely different, simpler angle. Keep it light.\n"
                    "• If they clearly refused (nahi chahiye, not interested, remove number) → thank them warmly, "
                    "output your [LEAD:...] tag, and append [CALL_END]."
                )
            elif self.bot_type == "recruitment":
                return (
                    f"The user just responded to your opening hook. Read their response carefully. {lang} "
                    "Do NOT use their name in this turn.\n\n"
                    "• If they're curious → move to STAGE 2 — QUALIFICATION & PITCH. Ask about their background.\n"
                    "• If hesitant or confused → clarify this isn't MLM or a job. Try a different angle.\n"
                    "• If clearly refused → thank them warmly, output [LEAD:...] tag, append [CALL_END]."
                )
            else:  # investment
                return (
                    f"The user just responded to your opening hook. Read their response carefully. {lang} "
                    "Do NOT use their name in this turn.\n\n"
                    "• If they engaged, asked questions, or showed interest → follow your S2 — DISCOVERY. "
                    "Understand how they manage finances. Ask one open question. Don't pitch yet.\n"
                    "• If they seem confused ('not clear', 'repeat', 'what?') → acknowledge warmly. "
                    "DON'T rephrase the same question. Try a completely different, simpler angle.\n"
                    "• If hesitant or skeptical ('why are you asking?', 'who is this?') → "
                    "follow your OBJECTION HANDLING. Acknowledge warmly. Try a different approach.\n"
                    "• If clearly refused (nahi chahiye, not interested, remove number, scam) → "
                    "respect their decision. Thank them warmly, output [LEAD:...] tag, append [CALL_END].\n\n"
                    "IMPORTANT: Never ask the same question you already asked, even in different words."
                )

        # ── QUALIFY ──────────────────────────────────────────────────
        elif self.state == CallState.QUALIFY:
            self.qualify_turns += 1

            # After 3+ turns of conversation, transition to appointment offer
            if self.qualify_turns >= 4:
                self.state = CallState.SCHEDULE
                if self.bot_type == "insurance":
                    return (
                        f"The conversation has been going well. {lang} "
                        "Do NOT use their name.\n\n"
                        "Read the user's last message:\n"
                        "• If they're still engaged → follow your MICRO-COMMITMENT stage. "
                        "Offer a free, independent review with संजीव सुराना. If they agree, ask which day works.\n"
                        "• If they pushed back or said no → follow your recovery system. Try one fresh angle.\n"
                        "• If they clearly refused → output [LEAD:...] tag and [CALL_END]."
                    )
                elif self.bot_type == "recruitment":
                    return (
                        f"Good conversation so far. {lang} "
                        "Do NOT use their name.\n\n"
                        "• If engaged → follow STAGE 4 — APPOINTMENT PITCH. "
                        "Suggest a meeting with Mr Sanjeev Surana. Ask which day works.\n"
                        "• If hesitant → try one more angle from your recovery system.\n"
                        "• If refused → output [LEAD:...] tag and [CALL_END]."
                    )
                else:  # investment
                    return (
                        f"Great conversation so far. {lang} "
                        "Do NOT use their name.\n\n"
                        "Read the user's last message:\n"
                        "• If they're engaged → follow your S5 — APPOINTMENT OFFER. "
                        "Transition naturally to offering a free, short chat with संजीव सुराना. "
                        "If they agree, ask which day works.\n"
                        "• If they pushed back → follow your S3.5 RECOVERY with a fresh angle.\n"
                        "• If clearly refused → output [LEAD:...] tag and [CALL_END]."
                    )

            # Still in discovery/curiosity building phase
            if self.bot_type == "insurance":
                return (
                    f"Continue the conversation naturally (turn {self.qualify_turns}/3 before appointment offer). {lang} "
                    "Do NOT use their name.\n\n"
                    "Read the user's last message:\n"
                    "• If they shared something → build on it. Follow your VALUE MOMENT or DISCOVERY stage.\n"
                    "• If they asked a question → answer it directly first, then ask ONE follow-up.\n"
                    "• If hesitant → gently share one relatable insight about health coverage gaps.\n"
                    "• If they refused → try one recovery angle. If hard refusal → [LEAD:...] + [CALL_END]."
                )
            elif self.bot_type == "recruitment":
                return (
                    f"Continue building rapport (turn {self.qualify_turns}/3). {lang} "
                    "Do NOT use their name.\n\n"
                    "• If engaged → continue STAGE 2 QUALIFICATION. Build on their response.\n"
                    "• If asked a question → answer directly, then ask one follow-up.\n"
                    "• If hesitant → share one insight about the opportunity.\n"
                    "• If refused → try recovery. Hard refusal → [LEAD:...] + [CALL_END]."
                )
            else:  # investment
                return (
                    f"Continue discovery naturally (turn {self.qualify_turns}/3 before appointment offer). {lang} "
                    "Do NOT use their name.\n\n"
                    "Read the user's last message:\n"
                    "• If they shared something → build on it. Follow your S2/S3 — DISCOVERY or CURIOSITY BUILDING.\n"
                    "• If they asked a question → answer it directly first, then ask ONE follow-up.\n"
                    "• If hesitant → follow S3.5 RECOVERY. Try a fresh angle based on what you know about them.\n"
                    "• If hard refusal (nahi chahiye, remove number) → output [LEAD:...] + [CALL_END].\n\n"
                    "IMPORTANT: Never repeat a question you already asked. Build on previous context."
                )

        # ── SCHEDULE ─────────────────────────────────────────────────
        elif self.state == CallState.SCHEDULE:
            return (
                f"You're scheduling a meeting with संजीव सुराना. {lang} "
                "Do NOT use their name.\n\n"
                "Read the user's last message:\n"
                "• If they gave a day and time → output: [APPOINTMENT: day=<YYYY-MM-DD>, time=<HH:MM>, "
                f"name={self.customer_name or 'unknown'}] then confirm warmly and append [CALL_END].\n"
                "• If they gave only a day → ask what time works — morning or evening?\n"
                "• If they gave only a time → ask which day — today, tomorrow, or another day?\n"
                "• If they agreed but didn't give day/time → ask warmly when would be convenient.\n"
                "• If they changed their mind → acknowledge, try one gentle recovery. "
                "If still no → output [LEAD:...] + [CALL_END].\n\n"
                "Today is " + datetime.now().strftime("%A, %d %B %Y") + "."
            )

        # ── CONFIRM ──────────────────────────────────────────────────
        elif self.state == CallState.CONFIRM:
            self.state = CallState.HANGUP
            return (
                f"Appointment confirmed: {self.scheduled_day} at {self.scheduled_time}. {lang} "
                f"Output: [APPOINTMENT: day={self.scheduled_day}, time={self.scheduled_time}, "
                f"name={self.customer_name or 'unknown'}] "
                "Confirm the details warmly, mention that संजीव सुराना will connect with them, "
                "output your [LEAD:...] tag, and append [CALL_END]."
            )

        # ── HANGUP ───────────────────────────────────────────────────
        else:
            return f"The call is ending. Say a warm, genuine goodbye. {lang} Output your [LEAD:...] tag and append [CALL_END]."