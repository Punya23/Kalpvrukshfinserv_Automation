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
        self.permission_attempts = 0  # Tracks recovery attempts in CHECK_PERMISSION
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

    def _lang_note(self) -> str:
        """Return a brief language reminder for injection into every turn instruction."""
        if self.language_preference == "english":
            return "(Speak in natural English only — no Hindi.)"
        return "(Speak in Hinglish — Hindi words in Devanagari, English terms in Latin.)"

    def _max_recovery(self) -> int:
        """Max recovery attempts before exit — matches the prompt's recovery system."""
        if self.bot_type == "investment":
            return 4   # Investment prompt S3.5 has 4 distinct recovery angles
        return 3       # Insurance and Recruitment have 3 recovery stages

    def get_instruction_for_current_state(
        self,
        user_text: str,
        classified_intent: str | None = None,
        schedule_info: dict | None = None,
    ) -> str:
        """
        The core navigator. Called once per turn, before the main LLM call.

        Philosophy: These instructions are director's notes — they tell the LLM
        what situation it's in and what the goal is, then let the prompt's stage
        descriptions (S1, S2, S3, etc.) guide the actual words.
        """
        self.total_turns += 1
        lang = self._lang_note()

        # Hard turn limit — cost control and troll protection
        if self.total_turns >= 15:
            self.state = CallState.HANGUP
            return f"The call has gone on too long. Thank them genuinely for their time and say goodbye warmly. {lang} Append [CALL_END]."
            
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
            self.state = CallState.CHECK_PERMISSION
            
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
            
            # Reference the prompt's S1/OPENING stage — let the LLM create the hook
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
        if self.state == CallState.CHECK_PERMISSION:
            if classified_intent == "YES":
                self.permission_attempts = 0
                self.state = CallState.QUALIFY
                if self.bot_type == "insurance":
                    return (
                        f"They engaged with your opening hook. {lang} "
                        "Follow your DISCOVERY stage — understand their current coverage situation. "
                        "Ask one open question to learn if they have a policy, corporate cover, or nothing. "
                        "Listen more than you talk."
                    )
                elif self.bot_type == "recruitment":
                    return (
                        f"They're curious. {lang} "
                        "Move to your STAGE 2 — QUALIFICATION & PITCH. "
                        "Match the pitch to their profession if you know it. "
                        "Ask one question to understand their background."
                    )
                else:  # investment
                    return (
                        f"They engaged positively. {lang} "
                        "Follow your S2 — DISCOVERY. Understand how they manage their finances. "
                        "Do they handle it themselves? Have existing investments? "
                        "Ask one open question. Don't pitch yet."
                    )
            elif classified_intent == "NO":
                self.permission_attempts += 1
                if self.permission_attempts >= 2:
                    # Second refusal → respect and exit
                    self.state = CallState.HANGUP
                    return (
                        f"They've refused twice. Respect their decision. {lang} "
                        "Thank them warmly for their time. "
                        "Output your [LEAD:...] tag with interest=dead, then append [CALL_END]."
                    )
                # First refusal → one recovery attempt (prompt says 'never exit on first refusal')
                if self.bot_type == "insurance":
                    return (
                        f"They pushed back on the opening. That's fine — don't take it personally. {lang} "
                        "Acknowledge their reaction warmly. Try a completely different angle: "
                        "mention that most people haven't reviewed their health coverage in years and "
                        "you just had a quick question. Keep it light and zero-pressure."
                    )
                elif self.bot_type == "recruitment":
                    return (
                        f"They weren't interested in the opening pitch. No problem. {lang} "
                        "Acknowledge naturally and try ONE different angle: "
                        "clarify this isn't a job or MLM — it's a partnership model. "
                        "Ask if they'd be open to just hearing one line about it."
                    )
                else:  # investment
                    return (
                        f"They pushed back or seem skeptical. That's normal. {lang} "
                        "Follow your OBJECTION HANDLING from the prompt. "
                        "Acknowledge their reaction warmly. Don't repeat the same hook. "
                        "Try a different angle — maybe mention that you just had a quick question, "
                        "no sales pitch, no product push. Keep it conversational."
                    )
            else:  # MAYBE
                self.permission_attempts += 1
                if self.permission_attempts >= 3:
                    # After 3 MAYBE attempts, they're clearly not engaging
                    self.state = CallState.HANGUP
                    return (
                        f"They've been hesitant multiple times. Respect their space. {lang} "
                        "Thank them warmly and offer to call another time. "
                        "Output your [LEAD:...] tag with interest=cold, then append [CALL_END]."
                    )
                if self.bot_type == "insurance":
                    return (
                        f"They seem hesitant — not a no, not a yes. {lang} "
                        "Don't pitch. Just ask for one minute of their time. "
                        "Mention that you have a quick question about their family's health protection."
                    )
                elif self.bot_type == "recruitment":
                    return (
                        f"They're unsure. {lang} "
                        "Don't push. Just say you have a quick question about business opportunities in their area. "
                        "Ask for just two minutes."
                    )
                else:
                    return (
                        f"They're hesitant but didn't refuse. {lang} "
                        "Acknowledge warmly. Make one gentle attempt — mention it's just a quick question "
                        "about making their savings work harder. Ask for two minutes."
                    )

        # ── QUALIFY ──────────────────────────────────────────────────
        elif self.state == CallState.QUALIFY:
            max_recovery = self._max_recovery()
            
            if classified_intent == "YES":
                self.recovery_count = 0
                self.qualify_turns += 1
                
                # Require at least 3 qualify turns before offering the consultation
                # This matches the prompt rule: "at least 3 organic conversational turns"
                if self.qualify_turns >= 3:
                    self.state = CallState.SCHEDULE
                    if self.bot_type == "insurance":
                        return (
                            f"The conversation has been good — they're engaged. {lang} "
                            "Follow your MICRO-COMMITMENT stage (Stage 6). "
                            "Offer the independent review framing first, then if they agree, "
                            "suggest scheduling with संजीव सुराना. Low-pressure transition."
                        )
                    elif self.bot_type == "recruitment":
                        return (
                            f"They're genuinely interested after good conversation. {lang} "
                            "Follow your STAGE 4 — APPOINTMENT PITCH. "
                            "Transition naturally to suggesting a meeting with Mr Sanjeev Surana. "
                            "Frame it as a partnership briefing, not a sales pitch."
                        )
                    else:  # investment
                        return (
                            f"Great discovery conversation. {lang} "
                            "Follow your S5 — APPOINTMENT OFFER. "
                            "Transition naturally to offering a free, short conversation with संजीव सुराना. "
                            "Generate a warm bridge — never make it sound transactional."
                        )
                
                # Still building — reference the appropriate prompt stage
                if self.bot_type == "insurance":
                    return (
                        f"Conversation flowing well (turn {self.qualify_turns}/3). {lang} "
                        "Follow your VALUE MOMENT or DISCOVERY stage. "
                        "Build on what they just shared. Ask one follow-up to go deeper. "
                        "If they mentioned a specific concern, explore it."
                    )
                elif self.bot_type == "recruitment":
                    return (
                        f"They're engaged (turn {self.qualify_turns}/3). {lang} "
                        "Continue your STAGE 2 — QUALIFICATION. "
                        "Build on their response. Share one relevant insight about the opportunity "
                        "and ask one question to understand their interest level better."
                    )
                else:  # investment
                    return (
                        f"Discovery going well (turn {self.qualify_turns}/3). {lang} "
                        "Follow your S2/S3 — DISCOVERY or CURIOSITY BUILDING. "
                        "Build on what they shared. Surface a blind spot or share a relatable insight. "
                        "Ask one follow-up that makes them think. Don't rush."
                    )
                    
            elif classified_intent == "NO":
                self.recovery_count += 1
                if self.recovery_count >= max_recovery:
                    self.state = CallState.HANGUP
                    return (
                        f"You've tried {max_recovery} recovery angles and they're still not interested. {lang} "
                        "Respect their decision. Thank them sincerely for their time. "
                        "Output your [LEAD:...] tag with the right status, then append [CALL_END]."
                    )
                
                # Reference the prompt's recovery system with specific angle guidance
                if self.bot_type == "insurance":
                    recovery_angles = [
                        "Follow your R1 recovery — mention that most people haven't reviewed their policy in years.",
                        "Follow your R2 recovery — connect to changing family responsibilities.",
                        "Follow your R3 recovery — frame it as an independent review with no obligations. Offer WhatsApp follow-up.",
                    ]
                elif self.bot_type == "recruitment":
                    recovery_angles = [
                        "Follow your STAGE 3 Attempt 1 — normalize their hesitation, mention recurring income model.",
                        "Follow your STAGE 3 Attempt 2 — clarify this isn't MLM or a job. It's a franchise model.",
                        "Follow your STAGE 3 Attempt 3 — last try. If still no, exit warmly.",
                    ]
                else:  # investment — 4 recovery angles
                    recovery_angles = [
                        "Follow your S3.5 Attempt 1 — normalize hesitation. Busy professionals often delay reviews.",
                        "Follow your S3.5 Attempt 2 — connect to a concrete milestone: children's education, home, retirement.",
                        "Follow your S3.5 Attempt 3 — social proof. Others in Pune found one conversation useful.",
                        "Follow your S3.5 Attempt 4 — zero-pressure framing. Free, short, no product push.",
                    ]
                
                angle_idx = min(self.recovery_count - 1, len(recovery_angles) - 1)
                return (
                    f"Soft refusal — recovery attempt {self.recovery_count}/{max_recovery}. {lang} "
                    f"{recovery_angles[angle_idx]} "
                    "Generate completely fresh language. Never repeat what you already said."
                )
                
            else:  # MAYBE
                if self.bot_type == "insurance":
                    return (
                        f"They're on the fence — not refusing, not agreeing. {lang} "
                        "Gently acknowledge their hesitation. Share one brief, relatable point "
                        "about how quickly medical costs are rising. Ask one soft follow-up."
                    )
                elif self.bot_type == "recruitment":
                    return (
                        f"They're thinking about it. {lang} "
                        "Don't push. Acknowledge naturally and share one small insight "
                        "about what professionals in similar roles are doing. Ask one gentle question."
                    )
                else:
                    return (
                        f"They're uncertain but still on the call. {lang} "
                        "Validate their hesitation empathetically. Share one brief relatable insight "
                        "based on what they've shared so far. Ask one gentle follow-up to keep them talking."
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
                    f"Both day ({self.scheduled_day}) and time ({self.scheduled_time}) confirmed. {lang} "
                    f"Output: [APPOINTMENT: day={self.scheduled_day}, time={self.scheduled_time}, "
                    f"name={self.customer_name or 'unknown'}] "
                    "Then confirm warmly and append [CALL_END]."
                )
            elif self.scheduled_day and not self.scheduled_time:
                return (
                    f"Day is set ({self.scheduled_day}) but no time yet. {lang} "
                    "Ask naturally what time works best for them — morning or afternoon/evening."
                )
            elif self.scheduled_time and not self.scheduled_day:
                return (
                    f"Time is set ({self.scheduled_time}) but no day yet. {lang} "
                    "Ask naturally which day works — today, tomorrow, or another day."
                )
            else:
                return (
                    f"They agreed to the meeting but haven't given day or time. {lang} "
                    "Ask warmly: which day and time would be convenient for them?"
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