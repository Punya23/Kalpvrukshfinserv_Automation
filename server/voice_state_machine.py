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
from server.config import config

# Post-call scoring/classification share the same OpenRouter→Groq fallback policy
# as the live conversation, via the shared llm_client module.
from server.llm_client import complete as _complete_with_fallback, message_content

logger = logging.getLogger(__name__)

# ── Utterance classifiers (deterministic — no LLM needed) ──────────

_GREETING_ONLY = re.compile(
    r"^(hello|hi|hey|hii|namaste|namaskar|haan|ha|hmm|hm|bolo|boliye|bolo na|"
    r"yes|yeah|yep|speak|sun|suniye|sun rahe|sun rahi|ok|okay|theek|thik|"
    r"हाँ|हां|जी|बोलिए|बोलो|नमस्ते|सुन|ठीक)\b",
    re.I,
)
_IDENTITY_QUESTION = re.compile(
    r"(kaun\b|who is|who are|kahan se|kis liye|kyun call|why.*call|aap kaun|"
    r"kon bol|konsa number|kis company|which company|kya kaam)",
    re.I,
)
_BUSY_REFUSAL = re.compile(
    r"(busy|meeting|abhi nahi|baad mein|baad me|call later|time nahi|not now|"
    r"can't talk|cannot talk|driving|driving me)",
    re.I,
)
_HARD_REFUSAL = re.compile(
    r"(nahi chahiye|not interested|remove number|call mat|don't call|mat karo|"
    r"no thanks|band karo|scam|fraud)",
    re.I,
)


def classify_utterance(text: str) -> str:
    """Classify a user utterance for smarter state navigation."""
    t = (text or "").strip()
    if not t:
        return "empty"
    tl = t.lower()
    if _HARD_REFUSAL.search(tl):
        return "hard_refusal"
    if _IDENTITY_QUESTION.search(tl):
        return "identity"
    if _BUSY_REFUSAL.search(tl):
        return "busy"
    words = tl.split()
    if _GREETING_ONLY.match(tl) or (len(words) <= 3 and any(
        w in tl for w in (
            "hello", "hi", "haan", "हाँ", "हां", "ji", "जी", "namaste", "नमस्ते",
            "boliye", "बोलिए", "bolo", "बोलो", "sun", "सुन",
        )
    )):
        return "greeting"
    return "substantive"


def _extract_json_object(raw: str) -> dict | None:
    """Best-effort JSON extraction when the LLM returns malformed JSON."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


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
        res = await _complete_with_fallback(
            [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=10
        )
        raw = message_content(res)
        if not raw:
            return "investment"
        result = re.sub(r"[^A-Z]", "", raw.upper())
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
        res = await _complete_with_fallback(
            [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=200
        )
        raw = re.sub(r"```json|```", "", message_content(res)).strip()
        if not raw:
            raise ValueError("empty LLM scoring response")
        parsed = _extract_json_object(raw)
        if not parsed:
            raise ValueError(f"unparseable scoring JSON: {raw[:80]}")
        
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

    def __init__(self, bot_type: str, customer_name: str = "", customer_category: str = "", agent_name: str = "Riya"):
        self.state             = CallState.OPENING
        self.bot_type          = bot_type
        self.customer_name     = customer_name or ""
        self.customer_category = customer_category or ""
        self.agent_name        = agent_name or "Riya"
        self.language_preference = "hinglish"  # Always Hinglish — never changes
        self.scheduled_day     = None
        self.scheduled_time    = None
        self.qualify_turns     = 0   # Turns spent in QUALIFY — gates appointment offer
        self.total_turns       = 0   # Hard limit guard
        self.recovery_count    = 0   # Track hard refusals/hesitations
        self.has_pivoted       = False # Ensure pivot only happens once
        self.welcome_interrupted = False  # User barged in during welcome
        self.bot_was_interrupted = False  # User barged in during bot reply
        self.welcome_spoken      = ""     # What was actually said in welcome (for context)
        self.chat_history      = []  # Trimmed to last 10 msgs for LLM context
        self.full_transcript   = []  # NEVER trimmed — used for scoring + disk logging
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
- Every Hindi word must be written in Devanagari. There are no exceptions. If you are unsure whether a word is Hindi, write it in Devanagari.
- Financial/English terms → Latin: SIP, mutual funds, savings, insurance, consultation, Kalpvruksh Finserv
- Founder name in Devanagari: संजीव सुराना — never "Sandeep Khurana" or any other name
- NEVER transliterate Hindi into Latin script (never write "main", "aap", "achha", "samajh", "sakti")
- NEVER mix scripts inside ONE word. The brand is ALWAYS full-Latin "Kalpvruksh Finserv" — never "कलpvruksh" or any Devanagari-Latin mash-up.

OUTPUT RULES:
- STRICT LENGTH: MAXIMUM 20 words per turn, in AT MOST 2 short sentences. Each sentence ≤ 12 words. NEVER exceed this — a long reply on a phone call sounds robotic and gets cut off.
- Talk like a real person on a call: short, casual, a fragment is fine. Use natural particles (हाँ, अरे, वैसे, बस, तो) instead of full formal clauses.
- BANNED robotic/AI phrases — NEVER say these or their Hindi versions (match YOUR OWN gender from the identity above): "I understand your concern", "I'd be happy to help", "rest assured", "great question", "as I mentioned", "feel free to", "at your convenience", "मैं आपकी बात पूरी तरह समझ सकती/सकता हूँ", "बिल्कुल सही कहा आपने". Do NOT open every turn with empathy or a summary of what they said. Use an empathy phrase like "समझ सकती/सकता हूँ" (your gender) AT MOST ONCE in the whole call.
- ENGLISH IS ONLY for the short 1-2 word reaction ("Got it", "Right"). EVERY question, answer, and goodbye stays in Hinglish/Hindi — NEVER a full English sentence (never "wishing you all the best", "have a great day"). Farewell in Hindi, e.g. "बहुत धन्यवाद, नमस्ते!".
- Openers: at least HALF your turns should start DIRECTLY with the Hindi content — no reaction word at all. When you DO react (fewer than half the turns), use ONE short clean ENGLISH word ("Sure", "Got it", "Right") and NEVER the same opener twice in a row. Never Hindi fillers like "अच्छा" or ANY spelling of "understood" ("समझ गयी", "समझ गई", "समझ गया"), never "..." (ellipses).
- Use SIMPLE everyday words. AVOID jargon: no "financial planning", "portfolio", "inflation", "strategy", "consultation", "aligned". Say it plainly (पैसे, बचत, महंगाई).
- NEVER ask where or how they keep their money, which bank, or about their FD/existing investments/policies — that feels intrusive, like prying. Sell the BENEFIT of your product instead; if THEY volunteer this info unprompted, you may acknowledge it warmly.
- One idea per turn. If they asked something, answer in ONE simple line — never lecture, never list.
- NEVER repeat the customer's name during the conversation. Name only at start or end.
- Sound like a smart, caring friend — never a telecaller reading a script.
- To end call: append [CALL_END] to your final farewell sentence and nowhere else.
- To confirm appointment: output [APPOINTMENT: day=<YYYY-MM-DD>, time=<HH:MM>, name=<name or unknown>]
  on its own line, then the farewell + [CALL_END].

{name_block}"""

        self.chat_history.append({"role": "system", "content": persona})

    # ── Helpers ──────────────────────────────────────────────────────

    def _hinglish_note(self) -> str:
        """Compact recency nudge injected into every turn instruction.

        The FULL rules (brevity, Hinglish script, jargon, no-AI-filler, etc.) already
        live in the system persona (chat_history[0]), which is resent on every API call —
        so restating them here in full would be pure duplication within the same call.
        This is intentionally short: it just re-anchors the single highest-failure-rate
        constraint (length) plus continuity, right next to the generation point, where a
        fast model pays the most attention.
        """
        return (
            "(≤20 words, ≤2 short sentences. No 'अच्छा' or any spelling of 'understood' "
            "('समझ गयी'/'समझ गई'/'समझ गया') as an opener, no ellipses. "
            "Build on what you just said, don't repeat or contradict your last line — "
            "and don't say 'समझ सकती/सकता हूँ' again if you already used it this call. Stay in Hinglish.)"
        )

    def get_turn_context(self, user_text: str) -> str:
        """Inject live call-awareness so the LLM doesn't act stateless after interruptions."""
        parts = []
        utterance = classify_utterance(user_text)

        if self.welcome_interrupted:
            parts.append(
                "LIVE CALL CONTEXT: You were interrupted mid-intro — the customer may NOT have heard "
                "your full opening. Do NOT assume they know why you called."
            )
        if self.bot_was_interrupted:
            parts.append(
                "LIVE CALL CONTEXT: The customer interrupted you while you were speaking. "
                "Respond to what THEY just said — do NOT repeat your previous sentence."
            )
        if utterance == "greeting":
            parts.append(
                f'The customer said "{user_text}" — a simple greeting/acknowledgement, NOT a deep answer. '
                "Acknowledge naturally ('हाँ जी!', 'जी बोलिए!'), then move ONE step forward. "
                "Do NOT launch into a credibility pitch or benefit as if they already engaged."
            )
        elif utterance == "identity":
            parts.append(
                f'The customer asked who you are / why you called: "{user_text}". '
                f'Answer clearly: you are {self.agent_name} from Kalpvruksh Finserv Pune + one-line reason. '
                "Do NOT pitch yet."
            )
        elif utterance == "busy":
            parts.append(
                f'The customer sounds busy: "{user_text}". '
                "Acknowledge, offer WhatsApp or callback later, keep it under 15 words."
            )
        elif utterance == "hard_refusal":
            parts.append(
                f'The customer refused: "{user_text}". '
                "Thank them warmly and append [CALL_END]. Do NOT argue or pitch."
            )

        if self.welcome_spoken:
            parts.append(f'Your welcome opener was: "{self.welcome_spoken[:120]}..."')

        return " ".join(parts)

    def _why_called_line(self) -> str:
        if self.bot_type == "insurance":
            return "आपके health cover पर एक ज़रूरी बात बतानी थी"
        if self.bot_type == "recruitment":
            return "एक extra income का आसान मौका बताना था"
        return "आपके पैसे बढ़ाने का एक आसान तरीका बताना था"

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

        # Track recoveries
        if "[RECOVERY]" in bot_text:
            self.recovery_count += 1
            bot_text = bot_text.replace("[RECOVERY]", "").strip()

        return bot_text

    # ── Core Navigator ──────────────────────────────────────────────

    def get_instruction_for_current_state(self, user_text: str) -> str:
        """
        The core navigator. Called once per turn, before the main LLM call.

        v4 philosophy: No language turn, no name turn. First user response
        after "Namaste" immediately gets intro + opening hook.
        """
        self.total_turns += 1
        lang = self._hinglish_note()
        utterance = classify_utterance(user_text)

        # Hard turn limit — cost control and troll protection
        if self.total_turns >= 15:
            self.state = CallState.HANGUP
            return f"The call has gone on too long. Thank them genuinely for their time and say goodbye warmly. {lang} Append [CALL_END]."

        # Global early exits — identity, busy, hard refusal override state
        if utterance == "hard_refusal":
            self.state = CallState.HANGUP
            return (
                f"The customer clearly refused: \"{user_text}\". {lang} "
                "Thank them warmly for their time. Do NOT argue. Append [CALL_END]."
            )
        if utterance == "identity":
            why = self._why_called_line()
            return (
                f'The customer asked who you are / why you called: "{user_text}". {lang} '
                f'Answer in ONE line: "जी, मैं {self.agent_name} बोल रही हूँ — Kalpvruksh Finserv, Pune से। {why}।" '
                "Do NOT pitch benefits yet. Do NOT repeat नमस्ते."
            )
        if utterance == "busy" and self.state in (CallState.OPENING, CallState.CHECK_PERMISSION):
            self.state = CallState.HANGUP
            return (
                f'The customer is busy: "{user_text}". {lang} '
                "Acknowledge warmly, say you'll call back or send details on WhatsApp. Append [CALL_END]."
            )

        # After repeated hesitation, stop pushing — but NEVER switch to another product/program.
        # Offer a low-pressure WhatsApp follow-up or callback, then bow out warmly.
        if self.recovery_count >= 3 and not self.has_pivoted:
            self.has_pivoted = True
            return (
                f"The customer has hesitated several times. {lang} "
                "Stop pushing. Do NOT switch to any other product, program, or income/partnership pitch. "
                "Warmly offer to send a few details on WhatsApp or to call another time — ONE short line. "
                "If they decline, thank them genuinely and append [CALL_END]."
            )

        # ── OPENING ─────────────────────────────────────────────────
        if self.state == CallState.OPENING:
            self.state = CallState.CHECK_PERMISSION

            # User barged in with just "hello" — finish intro naturally, don't jump to credibility pitch
            if utterance == "greeting" and self.welcome_interrupted:
                why = self._why_called_line()
                return (
                    f'The customer interrupted your intro and just said "{user_text}". {lang} '
                    f'Acknowledge warmly ("हाँ जी!" or "जी बोलिए!"), then finish your intro in ONE line: '
                    f'you are {self.agent_name} from Kalpvruksh Finserv Pune — {why}. '
                    'Ask if they have two minutes. Do NOT repeat नमस्ते. Do NOT launch into credibility stats yet.'
                )

            if self.bot_type == "insurance":
                credibility = "Kalpvruksh Finserv, Pune की trusted health-insurance advisory firm है — 2011 से, 500+ families का भरोसा"
            elif self.bot_type == "recruitment":
                credibility = "Kalpvruksh Finserv, Pune की leading advisory firm है — 2011 से, 500+ families के साथ"
            else:  # investment
                credibility = "Kalpvruksh Finserv, Pune की leading financial advisory firm है — 2011 से, 500+ families का भरोसा"

            return (
                f"You already greeted and said who you are and why you called. Do NOT introduce yourself again. {lang} "
                "React warmly in a couple words, then share ONE short credibility line to make them comfortable: "
                f"{credibility}. "
                "Do NOT ask a discovery question or pitch numbers yet — just build warmth and trust this turn. "
                "NEVER ask where/how they keep their money, which bank, or about their FD/existing investments — that feels intrusive. "
                "If they just said hello/hi/haan → acknowledge first, then credibility. "
                "If they say no/busy → offer to call later, then [CALL_END]. "
                "Keep it under 20 words. Simple words only."
            )

        # ── CHECK_PERMISSION ─────────────────────────────────────────
        if self.state == CallState.CHECK_PERMISSION:
            self.state = CallState.QUALIFY  # Optimistic advance

            # Greeting-only after credibility would feel dumb — treat as soft engagement
            if utterance == "greeting":
                return (
                    f'The customer responded with a simple acknowledgement: "{user_text}". {lang} '
                    "They heard you. Now pitch ONE concrete benefit with a real number/fact — "
                    "sell, don't interrogate. Do NOT repeat your credibility line."
                )

            if self.bot_type == "insurance":
                return (
                    f"The user just responded to your credibility line. Read their response carefully. {lang} "
                    "Do NOT use their name in this turn.\n\n"
                    "• If they engaged, asked questions, or showed curiosity → PITCH one concrete health-cover benefit "
                    "with a real fact/number (medical costs rise ~15% every year; cashless treatment at network hospitals; "
                    "one family floater covers everyone). Sell the benefit — do NOT ask about their current policy/coverage.\n"
                    "• If they seem confused, hesitant, or said 'not clear' → acknowledge warmly, DON'T repeat the same pitch. "
                    "Try a completely different benefit. Keep it light.\n"
                    "• If they clearly refused (nahi chahiye, not interested, remove number) → thank them warmly, "
                    "append [CALL_END]."
                )
            elif self.bot_type == "recruitment":
                return (
                    f"The user just responded to your credibility line. Read their response carefully. {lang} "
                    "Do NOT use their name in this turn.\n\n"
                    "• If they're curious → PITCH one concrete benefit of the partnership (recurring commission on renewals, "
                    "zero investment needed, full training provided). THEN, only if it flows naturally, ask about their profession.\n"
                    "• If hesitant or confused → clarify this isn't MLM or a job. Try a different benefit angle.\n"
                    "• If clearly refused → thank them warmly, append [CALL_END]."
                )
            else:  # investment
                return (
                    f"The user just responded to your credibility line. Read their response carefully. {lang} "
                    "Do NOT use their name in this turn.\n\n"
                    "• If they engaged, asked questions, or showed interest → PITCH one concrete SIP/mutual-fund benefit "
                    "with a real number (SIP mein 13-14% tak returns mil sakte hain, FD se kaafi zyada; ya power of "
                    "compounding — chhoti SIP se bada fund banta hai). Sell the benefit, don't interrogate.\n"
                    "• If they seem confused ('not clear', 'repeat', 'what?') → acknowledge warmly. "
                    "DON'T repeat the same pitch. Try a completely different benefit angle.\n"
                    "• If hesitant or skeptical ('why are you asking?', 'who is this?') → "
                    "follow your OBJECTION HANDLING. Acknowledge warmly. Try a different approach.\n"
                    "• If clearly refused (nahi chahiye, not interested, remove number, scam) → "
                    "respect their decision. Thank them warmly, append [CALL_END].\n\n"
                    "NEVER ask where/how they keep their money, which bank, or about their FD/existing investments. "
                    "IMPORTANT: Never pitch the same benefit twice, even in different words."
                )

        # ── QUALIFY ──────────────────────────────────────────────────
        elif self.state == CallState.QUALIFY:
            self.qualify_turns += 1

            # After 3+ turns of conversation, transition to appointment offer
            if self.qualify_turns >= 3:
                self.state = CallState.SCHEDULE
                if self.bot_type == "insurance":
                    return (
                        f"The conversation has been going well. {lang} "
                        "Do NOT use their name.\n\n"
                        "Read the user's last message:\n"
                        "- If they're still engaged → follow your MICRO-COMMITMENT stage. "
                        "Offer a free, independent review with हमारे Founder संजीव सुराना (15+ years experience). If they agree, ask which day works.\n"
                        "- If they're hesitant or pushed back → do NOT switch to any other product/program. "
                        "Gently offer to send details on WhatsApp or call another time. If they still decline, thank them warmly and append [CALL_END]."
                    )
                elif self.bot_type == "recruitment":
                    return (
                        f"Good conversation so far. {lang} "
                        "Do NOT use their name.\n\n"
                        "- If engaged → follow STAGE 4 — APPOINTMENT PITCH. "
                        "Suggest a meeting with हमारे Founder संजीव सुराना. Ask which day works.\n"
                        "• If hesitant → try one more angle from your recovery system.\n"
                        "• If refused → append [CALL_END]."
                    )
                else:  # investment
                    return (
                        f"Great conversation so far. {lang} "
                        "Do NOT use their name.\n\n"
                        "Read the user's last message:\n"
                        "- If they're engaged → follow your S5 — APPOINTMENT OFFER. "
                        "Transition naturally to offering a free, short chat with हमारे Founder संजीव सुराना (15+ years experience). "
                        "If they agree, ask which day works.\n"
                        "- If they're hesitant or pushed back → do NOT switch to any other product/program. "
                        "Gently offer to send details on WhatsApp or call another time. If they still decline, thank them warmly and append [CALL_END]."
                    )

            # Still in the benefit-selling phase — keep pitching, don't interrogate
            if self.bot_type == "insurance":
                return (
                    f"Continue the conversation naturally (turn {self.qualify_turns}/3 before appointment offer). {lang} "
                    "Do NOT use their name.\n\n"
                    "Read the user's last message:\n"
                    "• If they shared something or engaged → share ANOTHER concrete health-cover benefit/fact, "
                    "different from what you already used (rising medical costs, cashless network, family floater, quick claims).\n"
                    "• If they asked a question → answer it directly first, then add ONE more benefit.\n"
                    "• If hesitant → gently share one relatable insight about health coverage gaps.\n"
                    "• If they refused → try one recovery angle. If hard refusal → [CALL_END]."
                )
            elif self.bot_type == "recruitment":
                return (
                    f"Continue building rapport (turn {self.qualify_turns}/3). {lang} "
                    "Do NOT use their name.\n\n"
                    "• If engaged → share ANOTHER concrete benefit of the partnership, different from before "
                    "(recurring commission, zero investment, flexible hours, training/support).\n"
                    "• If asked a question → answer directly, then add one more benefit.\n"
                    "• If hesitant → share one insight about the opportunity.\n"
                    "• If refused → try recovery. Hard refusal → [CALL_END]."
                )
            else:  # investment
                return (
                    f"Continue the pitch naturally (turn {self.qualify_turns}/3 before appointment offer). {lang} "
                    "Do NOT use their name.\n\n"
                    "Read the user's last message:\n"
                    "• If they shared something or engaged → share ANOTHER concrete SIP/mutual-fund benefit, "
                    "different from what you already used (13-14% potential returns, power of compounding, महंगाई से आगे "
                    "निकलना, starting early, goal-based investing for kids' education or a home).\n"
                    "• If they asked a question → answer it directly first, then add ONE more benefit.\n"
                    "• If hesitant → follow S3.5 RECOVERY. Try a fresh benefit angle.\n"
                    "• If hard refusal (nahi chahiye, remove number) → [CALL_END].\n\n"
                    "NEVER ask where/how they keep their money, which bank, or about their FD/existing investments. "
                    "IMPORTANT: Never pitch the same benefit twice. Build on previous context."
                )

        # ── SCHEDULE ─────────────────────────────────────────────────
        elif self.state == CallState.SCHEDULE:
            return (
                f"You're scheduling a meeting with हमारे Founder संजीव सुराना. {lang} "
                "Do NOT use their name.\n\n"
                "Read the user's last message:\n"
                "• If they gave a day and time → output: [APPOINTMENT: day=<YYYY-MM-DD>, time=<HH:MM>, "
                f"name={self.customer_name or 'unknown'}] then confirm warmly and append [CALL_END].\n"
                "• If they gave only a day → ask what time works — morning or evening?\n"
                "• If they gave only a time → ask which day — today, tomorrow, or another day?\n"
                "• If they agreed but didn't give day/time → ask warmly when would be convenient.\n"
                "• If they said 'WhatsApp pe bhej do' or 'send details' or 'message kar do' → "
                "acknowledge warmly, say you'll send details on WhatsApp, and append [CALL_END].\n"
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
                "Confirm the details warmly, mention that हमारे Founder संजीव सुराना will connect with them, "
                "and append [CALL_END]."
            )

        # ── HANGUP ───────────────────────────────────────────────────
        else:
            return (
                f"The call is ending. Say a warm, genuine goodbye IN HINDI "
                f"(e.g. \"बहुत धन्यवाद, नमस्ते!\") — never an English sign-off. {lang} Append [CALL_END]."
            )