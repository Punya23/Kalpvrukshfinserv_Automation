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
    MAYBE = hesitant, busy but not refusing, asking for more info
    """
    prompt = (
        "You are classifying intent from a voice call.\n"
        "Output exactly one word: YES, NO, or MAYBE.\n\n"
        "YES  → agreement, interest, curiosity, willingness, questions (questions = interest)\n"
        "NO   → They clearly refused the pitch or showed hostility: nahi, no, not interested, timepass mat karo, scam hai, fraud\n"
        "MAYBE → hesitant, busy, unclear, asking for more info before deciding\n\n"
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


# ── Core State Machine ─────────────────────────────────────────────

class VoiceStateMachine:
    """
    One instance per call. Owns state, chat history, and scheduled appointment.

    Python drives transitions via get_instruction_for_current_state().
    The LLM only decides what words to speak — it never controls state.
    """

    def __init__(self, bot_type: str, customer_name: str = "", customer_category: str = ""):
        self.state             = CallState.CHECK_PERMISSION
        self.bot_type          = bot_type
        self.customer_name     = customer_name or ""
        self.customer_category = customer_category or ""
        self.language_preference = "hinglish"  # Default; updated after LANGUAGE_CHECK
        self.scheduled_day     = None
        self.scheduled_time    = None
        self.qualify_turns     = 0   # Turns spent in QUALIFY — gates the scheduling offer
        self.recovery_count    = 0   # Consecutive soft refusals in QUALIFY
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

    def get_instruction_for_current_state(
        self,
        user_text: str,
        classified_intent: str | None = None,
        schedule_info: dict | None = None,
    ) -> str:
        """
        The core navigator. Called once per turn, before the main LLM call.

        Responsibilities:
        1. Drive state transitions based on classified_intent and schedule_info.
        2. Return a brief situational instruction (injected as a system message)
           telling the LLM what situation it is in and what its goal is this turn.
        3. Enforce the hard turn limit.
        """
        self.total_turns += 1

        # Hard turn limit — cost control and troll protection
        if self.total_turns >= 15:
            self.state = CallState.HANGUP
            return "The call has reached its limit. Wrap up warmly and append [CALL_END]."
            
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
                "The user just responded to your initial 'Namaste'. "
                f"Introduce yourself: 'Main Kalpvruksh Finserv se {bot_name} {gender_form}.' "
                "Then ask their language preference naturally: 'क्या आप Hindi में बात करना prefer करेंगे या English में?' "
                "If they corrected their name, acknowledge the correction warmly before introducing yourself. "
                "Do NOT pitch anything yet. Just introduce + ask language."
            )

        # ── LANGUAGE_CHECK ───────────────────────────────────────────
        if self.state == CallState.LANGUAGE_CHECK:
            self.state = CallState.CHECK_PERMISSION
            
            # Language preference is set by the pipeline before this method is called
            lang = self.language_preference
            
            if lang == "english":
                lang_instruction = (
                    "LANGUAGE MODE: The customer chose ENGLISH. "
                    "Speak in clean, natural English. No Hindi words at all. "
                    "Keep it warm and conversational, not formal or corporate."
                )
            else:  # hinglish (default)
                lang_instruction = (
                    "LANGUAGE MODE: The customer chose HINDI. "
                    "Speak in natural Hinglish — 70% Hindi (Devanagari script), 30% English terms. "
                    "Hindi words in Devanagari: मैं, आप, अच्छा. English terms in Latin: SIP, insurance, mutual funds."
                )
            
            # Generate the hook based on bot type — LLM creates natural language
            if self.bot_type == "insurance":
                hook_instruction = (
                    "Now deliver your opening hook about healthcare protection. "
                    "The intent is to make them think about whether their current health coverage "
                    "is actually sufficient. Ask ONE engaging question. "
                    "Do NOT use fear-based language. Keep it consultative."
                )
            elif self.bot_type == "recruitment":
                hook_instruction = (
                    "Now deliver your opening hook about business partnership opportunities. "
                    "The intent is to create curiosity about an additional income vertical. "
                    "Ask ONE engaging question about their interest in growing their business."
                )
            else:  # investment
                hook_instruction = (
                    "Now deliver your opening hook about financial planning. "
                    "The intent is to make them pause and think about whether their savings "
                    "will be enough for rising future costs. Ask ONE engaging question."
                )
            
            return f"{lang_instruction}\n\n{hook_instruction}"

        # ── CHECK_PERMISSION ─────────────────────────────────────────
        if self.state == CallState.CHECK_PERMISSION:
            if classified_intent == "YES":
                self.state = CallState.QUALIFY
                return (
                    "The customer engaged positively with the opening hook. "
                    "Start discovery with one relevant open question based on their answer. "
                    "Do NOT pitch anything yet — just listen."
                )
            elif classified_intent == "NO":
                self.state = CallState.HANGUP
                return "The customer refused or indicated it is a wrong number. Apologize gracefully and naturally based on context, say goodbye warmly, and append [CALL_END]." 
            else:  # MAYBE
                return (
                    "The customer seems hesitant or was unclear. "
                    "Acknowledge their response warmly and make one more gentle attempt to get permission. "
                    "Do not pitch — just ask for two minutes of their time."
                )

        # ── QUALIFY ──────────────────────────────────────────────────
        elif self.state == CallState.QUALIFY:
            if classified_intent == "YES":
                self.recovery_count = 0
                self.qualify_turns += 1
                # Keep the conversation going for at least 2 qualify turns
                # before offering the consultation — builds engagement and call length
                if self.qualify_turns >= 2:
                    self.state = CallState.SCHEDULE
                    return (
                        "The customer is genuinely interested after good discovery. "
                        "Now transition naturally: introduce the free consultation with संजीव सुराना "
                        "and ask for a convenient day and time. Keep it warm and low-pressure."
                    )
                return (
                    f"The customer is engaged (qualify turn {self.qualify_turns}/2). "
                    "Ask ONE more discovery question to deepen the conversation. "
                    "Build curiosity with a relatable insight before pitching the consultation."
                )
            elif classified_intent == "NO":
                self.recovery_count += 1
                if self.recovery_count >= 3:
                    self.state = CallState.HANGUP
                    return (
                        "After three recovery attempts the customer is still not interested. "
                        "Thank them sincerely and append [CALL_END]."
                    )
                return (
                    f"Soft refusal — recovery attempt {self.recovery_count}/3. "
                    "Do NOT end the call. Use a completely fresh angle: a surprising insight, "
                    "a relatable story, or a new question. Never repeat what you already said."
                )
            else:  # MAYBE
                return (
                    "The customer is uncertain. Validate their hesitation empathetically, "
                    "share one brief relatable insight, and ask one gentle follow-up question. "
                    "Keep them talking — engagement is the goal."
                )

        # ── SCHEDULE ─────────────────────────────────────────────────
        elif self.state == CallState.SCHEDULE:
            day  = (schedule_info or {}).get("day")
            time = (schedule_info or {}).get("time")

            if day:
                self.scheduled_day = day
            if time:
                self.scheduled_time = time

            if self.scheduled_day and self.scheduled_time:
                self.state = CallState.CONFIRM
                return (
                    f"Both day ({self.scheduled_day}) and time ({self.scheduled_time}) are confirmed. "
                    f"Output: [APPOINTMENT: day={self.scheduled_day}, time={self.scheduled_time}, "
                    f"name={self.customer_name or 'unknown'}] "
                    "Then confirm the slot warmly and append [CALL_END]."
                )
            elif self.scheduled_day and not self.scheduled_time:
                return (
                    f"Day is set ({self.scheduled_day}) but time is missing. "
                    "Ask specifically: kis samay convenient rahega?"
                )
            elif self.scheduled_time and not self.scheduled_day:
                return (
                    f"Time is set ({self.scheduled_time}) but day is missing. "
                    "Ask: aaj ya kal — kab better rahega?"
                )
            else:
                return (
                    "Neither day nor time collected yet. "
                    "Ask clearly and simply: aap kab free honge — aur kis samay?"
                )

        # ── CONFIRM ──────────────────────────────────────────────────
        elif self.state == CallState.CONFIRM:
            self.state = CallState.HANGUP
            return (
                f"Appointment confirmed: {self.scheduled_day} at {self.scheduled_time}. "
                f"Output: [APPOINTMENT: day={self.scheduled_day}, time={self.scheduled_time}, "
                f"name={self.customer_name or 'unknown'}] "
                "Then say a warm, genuine goodbye and append [CALL_END]."
            )

        # ── HANGUP ───────────────────────────────────────────────────
        else:
            return "Wrap up warmly and append [CALL_END]."