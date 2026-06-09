"""
Kalpvruksh Finserv — Voice Call State Machine

Philosophy: Python is the navigator. The LLM is the speaker.
- State transitions are deterministic Python — never delegated to the LLM.
- Per-turn situational instructions tell the LLM exactly what situation it is in
  and what its goal is for this turn, without bloating the system prompt.
- v4: No language selection. No name asking. LLM-based lead scoring.
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
    OPENING          = "OPENING"          # Intro + opening hook in ONE turn
    CHECK_PERMISSION = "CHECK_PERMISSION" # Hook delivered; waiting for permission to continue
    QUALIFY          = "QUALIFY"           # Discovery and curiosity building
    SCHEDULE         = "SCHEDULE"          # Collecting day + time for Sanjeev sir's callback
    CONFIRM          = "CONFIRM"          # Both day + time received; confirming appointment
    HANGUP           = "HANGUP"           # Final state; call ending

ALLOWED_TRANSITIONS = {
    "OPENING":          {"CHECK_PERMISSION", "HANGUP"},
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


async def score_lead_with_llm(transcript: str, bot_type: str = "investment") -> dict:
    """
    LLM-based lead scoring. Called AFTER the call ends (no customer-facing latency).
    Sends the full transcript to the LLM and gets back a structured score.
    
    Returns: {
        "score": 0-10,
        "category": "HOT" | "WARM" | "COLD" | "DNC",
        "interest": "hot" | "warm" | "cold" | "dead",
        "objection": "none" | "busy" | "has_advisor" | "not_interested" | "send_details",
        "appointment": "yes" | "no",
        "summary": "one-line summary of what happened"
    }
    """
    prompt = f"""You are scoring a sales call for a financial advisory firm (Kalpvruksh Finserv, Pune).
Read the transcript below and output ONLY valid JSON — no markdown, no explanation.

SCORING RULES:
- Score 8-10 (HOT): Appointment booked, or customer showed strong interest and asked multiple questions, or agreed to a meeting
- Score 5-7 (WARM): Customer engaged in conversation, asked questions, showed some curiosity, but didn't commit. OR was busy but not hostile. OR said "send details" or "call later"
- Score 2-4 (COLD): Customer was uninterested but not hostile. Short call, minimal engagement. Said "socha nahi" or "dekhenge" but wasn't rude.
- Score 0-1 (DNC): Customer explicitly said "remove number", was angry/abusive, or said "call mat karna". Do Not Contact.

IMPORTANT: If the customer ENGAGED in conversation (asked questions, gave responses, discussed their finances), they are AT LEAST WARM (score 5+) even if no appointment was booked.
A call that ended due to technical issues or latency should be scored WARM (5), not COLD.
A customer who was curious but hesitant is WARM (5-6), not COLD.

Bot type: {bot_type}

TRANSCRIPT:
{transcript}

Output JSON:
{{"score": <0-10>, "category": "<HOT|WARM|COLD|DNC>", "interest": "<hot|warm|cold|dead>", "objection": "<none|busy|has_advisor|not_interested|send_details>", "appointment": "<yes|no>", "summary": "<one line summary>"}}"""

    try:
        res = await groq_client.chat.completions.create(
            model=CLASSIFIER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        raw = re.sub(r"```json|```", "", res.choices[0].message.content.strip()).strip()
        parsed = json.loads(raw)
        
        # Validate and clamp score
        score = max(0, min(10, int(parsed.get("score", 0))))
        category = parsed.get("category", "COLD").upper()
        if category not in ("HOT", "WARM", "COLD", "DNC"):
            category = "COLD"
        
        return {
            "score": score,
            "category": category,
            "interest": parsed.get("interest", "cold"),
            "objection": parsed.get("objection", "none"),
            "appointment": parsed.get("appointment", "no"),
            "summary": parsed.get("summary", "")[:200],
        }
    except Exception as e:
        logger.error(f"score_lead_with_llm error: {e}")
        return {
            "score": 3,
            "category": "COLD",
            "interest": "cold",
            "objection": "none",
            "appointment": "no",
            "summary": "Scoring failed — defaulted to COLD",
        }


class VoiceStateMachine:
    """
    One instance per call. Owns state, chat history, and scheduled appointment.

    v4 — Production-ready:
    - NO language selection (always Hinglish)
    - NO name asking (name from database)
    - LLM-based lead scoring (post-call)
    - Intro + hook in ONE turn (saves a full round-trip)
    """

    def __init__(self, bot_type: str, customer_name: str = "", customer_category: str = ""):
        self.state             = CallState.OPENING
        self.bot_type          = bot_type
        self.customer_name     = customer_name or ""
        self.customer_category = customer_category or ""
        self.language_preference = "hinglish"  # Always Hinglish — never changes
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

        # Name block — explicit that name is KNOWN from database
        if self.customer_name.strip():
            name_block = (
                f'CUSTOMER NAME FROM DATABASE: "{self.customer_name}"\n'
                'You already greeted them by name. Do NOT ask for their name again.\n'
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

    def _hinglish_note(self) -> str:
        """Brief Hinglish reminder injected into every turn instruction."""
        return "(Speak in Hinglish — Hindi words in Devanagari, English financial terms in Latin. NEVER pure Hindi.)"

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

        v4 philosophy: No language turn, no name turn. First user response
        after "Namaste" immediately gets intro + opening hook.
        """
        self.total_turns += 1
        lang = self._hinglish_note()

        # Hard turn limit — cost control and troll protection
        if self.total_turns >= 15:
            self.state = CallState.HANGUP
            return f"The call has gone on too long. Thank them genuinely for their time and say goodbye warmly. {lang} Append [CALL_END]."

        # ── OPENING ─────────────────────────────────────────────────
        # User just responded to "Namaste Sanjeev ji?" — introduce yourself AND deliver hook in ONE turn
        if self.state == CallState.OPENING:
            self.state = CallState.CHECK_PERMISSION

            if self.bot_type == "insurance":
                bot_name = "Aarav"
                gender_form = "bol raha hoon"
                hook = (
                    "Then immediately follow your OPENING & DISCOVERY stage. "
                    "Ask one engaging question about their healthcare protection. "
                    "Be consultative, not salesy. Make them think."
                )
            elif self.bot_type == "recruitment":
                bot_name = "Riya"
                gender_form = "bol rahi hoon"
                hook = (
                    "Then immediately follow your STAGE 1 — BUSINESS HOOK. "
                    "Create curiosity about the business partnership opportunity. "
                    "Generate a fresh, natural opening."
                )
            else:  # investment
                bot_name = "Riya"
                gender_form = "bol rahi hoon"
                hook = (
                    "Then immediately follow your S1 — OPENING. "
                    "Make them think about whether their savings strategy is aligned with rising future costs. "
                    "Generate a fresh question — never the same phrasing twice. "
                    "NEVER ask about specific amounts, income, salary, or portfolio value."
                )

            return (
                f"The user just responded to your 'Namaste'. {lang} "
                f"Introduce yourself briefly: 'Main Kalpvruksh Finserv se {bot_name} {gender_form}.' "
                f"If they corrected their name, acknowledge it warmly first. "
                f"{hook}"
            )

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
                    "append [CALL_END]."
                )
            elif self.bot_type == "recruitment":
                return (
                    f"The user just responded to your opening hook. Read their response carefully. {lang} "
                    "Do NOT use their name in this turn.\n\n"
                    "• If they're curious → move to STAGE 2 — QUALIFICATION & PITCH. Ask about their background.\n"
                    "• If hesitant or confused → clarify this isn't MLM or a job. Try a different angle.\n"
                    "• If clearly refused → thank them warmly, append [CALL_END]."
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
                    "respect their decision. Thank them warmly, append [CALL_END].\n\n"
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
                        "• If they clearly refused → append [CALL_END]."
                    )
                elif self.bot_type == "recruitment":
                    return (
                        f"Good conversation so far. {lang} "
                        "Do NOT use their name.\n\n"
                        "• If engaged → follow STAGE 4 — APPOINTMENT PITCH. "
                        "Suggest a meeting with Mr Sanjeev Surana. Ask which day works.\n"
                        "• If hesitant → try one more angle from your recovery system.\n"
                        "• If refused → append [CALL_END]."
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
                        "• If clearly refused → append [CALL_END]."
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
                    "• If they refused → try one recovery angle. If hard refusal → [CALL_END]."
                )
            elif self.bot_type == "recruitment":
                return (
                    f"Continue building rapport (turn {self.qualify_turns}/3). {lang} "
                    "Do NOT use their name.\n\n"
                    "• If engaged → continue STAGE 2 QUALIFICATION. Build on their response.\n"
                    "• If asked a question → answer directly, then ask one follow-up.\n"
                    "• If hesitant → share one insight about the opportunity.\n"
                    "• If refused → try recovery. Hard refusal → [CALL_END]."
                )
            else:  # investment
                return (
                    f"Continue discovery naturally (turn {self.qualify_turns}/3 before appointment offer). {lang} "
                    "Do NOT use their name.\n\n"
                    "Read the user's last message:\n"
                    "• If they shared something → build on it. Follow your S2/S3 — DISCOVERY or CURIOSITY BUILDING.\n"
                    "• If they asked a question → answer it directly first, then ask ONE follow-up.\n"
                    "• If hesitant → follow S3.5 RECOVERY. Try a fresh angle based on what you know about them.\n"
                    "• If hard refusal (nahi chahiye, remove number) → [CALL_END].\n\n"
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
                "If still no → append [CALL_END].\n\n"
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
                "and append [CALL_END]."
            )

        # ── HANGUP ───────────────────────────────────────────────────
        else:
            return f"The call is ending. Say a warm, genuine goodbye. {lang} Append [CALL_END]."