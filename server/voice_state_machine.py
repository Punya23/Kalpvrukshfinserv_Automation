"""
Kalpvruksh Finserv — Voice Call State Machine

Philosophy: The LLM is the product. We use a capable 70B model and invest in
excellent prompts rather than compensating for a weak model with hardcoded
fallbacks, keyword dictionaries, and regex patterns.

Zero hardcoding. Zero manual fallbacks. Trust the model.
"""

import re
import json
import logging
from enum import Enum
from groq import AsyncGroq
from server.config import config

logger = logging.getLogger(__name__)

# Initialize Groq client
groq_client = AsyncGroq(api_key=config.GROQ_API_KEY)

# ---------------------------------------------------------------
# Model Selection
# ---------------------------------------------------------------
# Use the best model for EVERYTHING. The 70B model understands Hindi,
# Hinglish, conversational nuance, and edge cases natively.
# On Groq's infrastructure, 70B adds ~200-300ms vs 8B — negligible
# for a voice call that already spends 1-2s on TTS.
# ---------------------------------------------------------------
CLASSIFIER_MODEL = "llama-3.3-70b-versatile"
CONVERSATION_MODEL = config.LLM_MODEL or "llama-3.3-70b-versatile"


class CallState(Enum):
    GREETING = "GREETING"
    CHECK_PERMISSION = "CHECK_PERMISSION"
    QUALIFY = "QUALIFY"
    SCHEDULE = "SCHEDULE"
    CONFIRM = "CONFIRM"
    HANGUP = "HANGUP"


class VoiceStateMachine:
    """
    Manages conversation flow state. Uses descriptive situation-based
    instructions rather than prescriptive scripts. The LLM decides
    what to say — we just tell it what's happening and what the goal is.
    """

    def __init__(self, bot_type: str, customer_name: str = "", customer_category: str = ""):
        self.state = CallState.GREETING
        self.bot_type = bot_type  # "investment", "insurance", "recruitment"
        self.customer_name = customer_name or ""
        self.customer_category = customer_category or ""
        self.qualify_turns = 0
        self.scheduled_day = None
        self.scheduled_time = None
        self.chat_history = []
        self.total_turns = 0

        self._initialize_persona()

    def _initialize_persona(self):
        """
        Build a comprehensive persona prompt optimized for spoken output.
        
        Key insight: Polly Kajal is a Hindi TTS (hi-IN). When it sees Roman-script
        Hindi like "main", it reads it as the English word "main". By writing Hindi
        in Devanagari (मैं), Polly pronounces it correctly. This is the logical fix —
        match the output format to the TTS engine's expected input.
        """

        # --- Identity by bot type ---
        if self.bot_type == "insurance":
            identity_block = (
                "You are Aditi, a warm and genuinely caring health insurance advisor "
                "at कल्पवृक्ष Finserv, Pune (partnered with Star Health). "
                "You help families understand the importance of health coverage. "
                "You know about Star Health plans, family floater policies, and claim processes."
            )
        elif self.bot_type == "recruitment":
            identity_block = (
                "You are Riya, a warm and professional recruitment specialist "
                "at कल्पवृक्ष Finserv, Pune. You call independent insurance/loan agents "
                "and financial advisors to invite them to join as Wealth Partners. "
                "You offer access to 30+ mutual fund houses, Star Health, automated HNI "
                "lead generation, and competitive payouts."
            )
        else:  # investment (default)
            identity_block = (
                "You are Riya, a warm, friendly, and genuinely helpful financial advisor "
                "at कल्पवृक्ष Finserv, Pune. You help people understand how to grow their "
                "savings beyond traditional FDs and savings accounts."
            )

        # --- Name handling — let the LLM use its judgment ---
        if self.customer_name.strip():
            name_block = (
                f'CUSTOMER NAME FROM OUR DATABASE: "{self.customer_name}"\n'
                "Use your judgment when addressing the customer:\n"
                "- If this is a valid person's name, address them warmly using it.\n"
                "- If it is not a valid person's name (e.g. a business category/name, a placeholder, or a generic term), "
                'just address them respectfully using "आप" without using any name.'
            )
        else:
            name_block = (
                "The customer's name is not known. Do NOT guess or make up a name. "
                'Use respectful "आप" language throughout.'
            )

        # --- Customer context — gives the LLM profession/category info ---
        if self.customer_category.strip():
            name_block += (
                f'\n\nCUSTOMER CONTEXT: This person is from the "{self.customer_category}" category in our database. '
                "Use this context naturally in your conversation — for example, relate your offering to their profession. "
                "Do NOT awkwardly mention their category verbatim; weave it in naturally."
            )

        # --- Build the full persona ---
        persona = f"""{identity_block}

WRITING FORMAT — THIS IS CRITICAL:
Your text will be read aloud by a Hindi text-to-speech engine (AWS Polly, Kajal voice, hi-IN).
You MUST follow this writing format for correct pronunciation:

1. Write ALL Hindi/Hinglish words in DEVANAGARI SCRIPT:
   मैं, बोल रही हूँ, आप, अच्छा, देखिए, बताइए, हाँ, नहीं, बिल्कुल, ठीक है, कल, परसों
2. Write ALL English words in ENGLISH (Latin) script:
   savings, account, mutual funds, investment, consultation, FD, SIP, returns
3. This creates natural code-mixed text. Examples:
   ✅ "अच्छा, actually आपकी savings कहाँ invest हैं?"
   ✅ "बिल्कुल! संजीव सुराना sir आपको personally guide करेंगे।"
   ✅ "मैं समझती हूँ, आपके लिए एक free consultation arrange कर देती हूँ।"
4. NEVER write Hindi words in Roman/Latin script. The TTS will mispronounce them:
   ❌ "main bol rahi hoon" → TTS reads "main" as English word "MAIN"
   ❌ "achha" → TTS doesn't know this is अच्छा
   ❌ "Surana" → TTS may mispronounce as "Khurana"
   ✅ Always write: मैं बोल रही हूँ, अच्छा, सुराना
5. Proper nouns in Devanagari: संजीव सुराना sir, कल्पवृक्ष Finserv
6. STRICT NAME RULE: The founder's name is EXACTLY संजीव सुराना (Sanjeev Surana). NEVER call him "संदीप खुराना" (Sandeep Khurana).
7. Use correct Hinglish verb forms. Never say "invest है" — say "invested हैं" or "invest किया है". Apply this to all English verbs used with Hindi grammar.

PERSONALITY & SPOKEN NATURALNESS:
- You are on a PHONE CALL. Your text will be spoken aloud, not read on a screen.
- Sound like a smart, caring friend who knows about finance — NOT a telecaller or IVR system.
- Start responses with a natural reaction to what they said, not a rehearsed line.
- Add brief pauses with "..." between thoughts for natural rhythm:
  "अच्छा... तो आपकी savings mostly FD में हैं?"
- Use warm expressions naturally: "बिल्कुल!", "जी, बहुत अच्छे", "सही कहा!"
- DO NOT use over-enthusiastic words like "वाह!" (Wow!), especially when scheduling a call or doing logistical tasks.
- Vary your openings — don't always start with the same word.
- When acknowledging, sound genuinely interested, not mechanical.

GENDER GRAMMAR — YOU ARE FEMALE:
In ALL Hindi portions, ALWAYS use feminine verb forms:
✅ "मैं बोल रही हूँ", "मैं चाहती हूँ", "मैं बताती हूँ", "मैंने सोचा", "मैं समझती हूँ", "मैं कर सकती हूँ"
❌ NEVER: "मैं बोल रहा हूँ", "मैं चाहता हूँ", "मैं बताता हूँ" — these are masculine forms.

{name_block}

CONVERSATION RULES:
1. Keep EVERY response to 1-2 sentences maximum. Then STOP and let them speak.
2. Ask only ONE question at a time. Never stack multiple questions.
3. Mirror their energy — if they're enthusiastic, be enthusiastic. If cautious, be gentle.
4. NEVER fabricate information, promise specific returns, quote percentages, or recommend specific funds/plans.
5. NEVER argue with the customer, even if they say something factually wrong.
6. If someone is angry, hostile, or uses harsh language — immediately apologize warmly, thank them, and end the call.
7. Your job is ONLY to qualify the lead and book a callback with संजीव सुराना sir. You do NOT close deals.

COMPANY CONTEXT:
- Company: कल्पवृक्ष Finserv, Pune (15 years in business)
- Senior Advisor: संजीव सुराना sir (founder, 15+ years experience) — this is who you schedule callbacks with
- Your role: Qualify leads → Schedule callback with संजीव sir → End call

THE [CALL_END] TAG:
When the conversation is truly ending — they said bye, refused, got angry, or you've confirmed a callback — append [CALL_END] at the very end.
Examples:
- "धन्यवाद मुझसे बात करने के लिए, आपका दिन शुभ हो। [CALL_END]"
- "संजीव sir कल पाँच बजे call करेंगे। धन्यवाद मुझसे बात करने के लिए, आपका दिन शुभ हो। [CALL_END]"
NEVER use [CALL_END] while you are still asking questions or waiting for their response."""

        self.chat_history.append({"role": "system", "content": persona})

    def get_instruction_for_current_state(
        self,
        user_input: str,
        classified_intent: str = None,
        schedule_info: dict = None,
    ) -> str:
        """
        Generate situational context for the LLM based on conversation state.

        These instructions describe WHAT IS HAPPENING and WHAT THE GOAL IS.
        They do NOT prescribe exact words. The LLM decides how to speak naturally.
        """
        self.total_turns += 1

        # Safety valve: prevent infinite conversations
        if self.total_turns >= 15:
            self.state = CallState.HANGUP
            return (
                "This conversation has been going on for a while. "
                "It's time to wrap up gracefully. Thank the customer for their time, "
                "let them know संजीव sir will reach out soon, and say a warm goodbye. "
                "Include [CALL_END] at the end."
            )

        # ----- CHECK_PERMISSION state -----
        if self.state == CallState.CHECK_PERMISSION:
            if classified_intent == "NO":
                self.state = CallState.HANGUP
                return (
                    "The customer is not interested or declined to talk. "
                    "Gracefully apologize for the disturbance, thank them, and end the call warmly. "
                    "No pushing, no convincing, no follow-up questions. "
                    "Include [CALL_END] at the end."
                )
            elif classified_intent == "BUSY":
                self.state = CallState.SCHEDULE
                return (
                    "The customer sounds busy right now but hasn't refused outright. "
                    "Briefly and warmly ask when they'd be free for a callback. "
                    "STRICT RULE: Do NOT suggest, list, or offer any specific days or time slots. "
                    "Simply ask 'आप कब free होंगे?' and STOP. Let them answer. "
                    "Do NOT include [CALL_END]."
                )
            else:  # YES or ambiguous → give benefit of the doubt
                self.state = CallState.QUALIFY
                if self.bot_type == "insurance":
                    return (
                        "The customer agreed to talk! Start qualifying. "
                        "Ask about their current health insurance situation — do they have family coverage, "
                        "or are they exploring options? One natural question."
                    )
                elif self.bot_type == "recruitment":
                    return (
                        "The customer agreed to talk! Start qualifying. "
                        "Ask if they're currently working as an insurance or financial advisor, "
                        "or looking for new income opportunities. One natural question."
                    )
                else:  # investment
                    return (
                        "The customer agreed to talk! Start qualifying by asking where their savings "
                        "currently are — FD, savings account, mutual funds, or something else. "
                        "This gets them talking about their situation. One natural question."
                    )

        # ----- QUALIFY state -----
        elif self.state == CallState.QUALIFY:
            if classified_intent == "NO":
                self.qualify_turns += 1
                return (
                    "The customer explicitly stated they already have investments/insurance or are not interested. "
                    "DO NOT blindly pitch. Acknowledge their situation and gracefully ask if they'd be open to a casual, no-obligation portfolio review with संजीव sir instead. "
                    "1-2 sentences max."
                )

            if self.qualify_turns == 0:
                self.qualify_turns += 1
                if self.bot_type == "insurance":
                    return (
                        "The customer just shared their insurance situation. "
                        "Acknowledge what they said naturally, then pivot to offering a free "
                        "consultation with senior advisor संजीव सुराना sir for a personalized review. "
                        "Ask if they'd be open to a brief call from him. 1-2 sentences max."
                    )
                elif self.bot_type == "recruitment":
                    return (
                        "The customer shared their professional background. "
                        "Acknowledge it warmly, then briefly pitch the कल्पवृक्ष partnership opportunity "
                        "(30+ brands, leads, payouts). Suggest a meeting with संजीव सुराना sir. "
                        "1-2 sentences max."
                    )
                else:  # investment
                    return (
                        "The customer told you about their savings. "
                        "Acknowledge what they said naturally, then mention that there are "
                        "options that could work better for them. Offer a free consultation by first introducing संजीव सुराना sir — mention that he is the founder of Kalpavriksha Finserv with 15+ years of experience in wealth management. Then ask if you can schedule a free call with him personally. Example: \"हमारे साथ संजीव सुराना sir हैं — वो इस field में 15 साल से हैं और कल्पवृक्ष Finserv के founder हैं। वो personally आपको एक free consultation देना चाहते हैं। क्या मैं उनसे आपकी एक short call schedule करवा सकती हूँ?\""
                    )
            else:
                # This is their response to "Can I schedule a consultation?"
                if classified_intent == "NO":
                    self.state = CallState.HANGUP
                    return (
                        "The customer declined the consultation offer. "
                        "Respond gracefully — thank them, wish them well, end warmly. "
                        "No pushing. Include [CALL_END] at the end."
                    )
                else:
                    self.state = CallState.SCHEDULE
                    return (
                        "STRICT INSTRUCTION: The user has agreed to a callback but has NOT given a day or time yet. You MUST ask for their availability before confirming anything. Your response must be ONLY: \"ठीक है जी! आप किस दिन और किस समय free होंगे?\" — Do NOT suggest a day. Do NOT suggest a time. Do NOT say \"kal\" or \"5 baje\" or any specific slot. Just ask the open question and wait."
                    )

        # ----- SCHEDULE state -----
        elif self.state == CallState.SCHEDULE:
            if schedule_info and schedule_info.get("status") == "TimeGiven":
                self.state = CallState.CONFIRM
                self.scheduled_day = schedule_info.get("day")
                self.scheduled_time = schedule_info.get("time")

                meet_word = "meeting" if self.bot_type == "recruitment" else "call"
                return (
                    f"The customer wants the callback on {self.scheduled_day} at {self.scheduled_time}. "
                    f"Confirm that संजीव sir will {meet_word} them at this time. "
                    "Thank them warmly and wish them a great day. "
                    "Include [CALL_END] at the end."
                )
            else:
                # They didn't give a clear day/time
                return (
                    "The customer's response didn't contain a clear day or time. "
                    "Gently ask them again — when would be convenient for the callback? "
                    "Let them choose. One natural sentence. Do NOT include [CALL_END]."
                )

        # ----- CONFIRM / HANGUP state -----
        elif self.state in (CallState.CONFIRM, CallState.HANGUP):
            return (
                "The conversation is ending. Say a warm goodbye. "
                "Include [CALL_END] at the end."
            )

        # ----- Fallback -----
        return "Continue the conversation naturally. Be helpful, warm, and brief."


# ===================================================================
# INTELLIGENT LLM CLASSIFIERS
#
# These use the 70B model with carefully crafted prompts.
# NO hardcoded fallbacks. NO keyword dictionaries. NO regex parsing.
# The model understands Hindi/Hinglish natively — trust it.
# ===================================================================

async def classify_permission(user_input: str) -> str:
    """
    Determine if the customer agreed to talk, refused, or is busy.

    Uses 70B model at temperature=0.0 for deterministic classification.
    No keyword fallback — the model handles all languages and edge cases.
    """
    prompt = (
        "You are classifying a customer's response during a phone call.\n\n"
        "CONTEXT: An AI assistant just introduced herself and asked the customer "
        "if they can spare 30 seconds. The customer responded with the text below.\n\n"
        "Classify their response as exactly ONE of these three categories:\n\n"
        "YES — They agreed, showed interest, gave permission, or responded with any "
        "affirmative/neutral greeting. This includes:\n"
        "  - Direct agreement: haan, ok, sure, yes, theek hai, chalo, boliye\n"
        "  - Engagement: bolo, batao, sunao, karo baat, bataiye\n"
        "  - Greetings that imply willingness: hello, hi, ji, ji boliye, haan ji\n"
        "  - Ambiguous/unclear/noisy responses (give benefit of the doubt)\n"
        "  - Any single word or short response that isn't a clear refusal\n\n"
        "NO — They clearly and explicitly refused, showed hostility, or stated they already have the product:\n"
        "  - Direct refusal: nahi, no, not interested, nahi chahiye\n"
        "  - Already have it: pehle se hai, already done, sorted, nahi chahiye, already have mutual funds\n"
        "  - Hostility: timepass mat karo, scam hai, fraud, bakwas, phone mat karo\n"
        "  - Explicit dismissal: don't call, koi zaroorat nahi, band karo\n\n"
        "BUSY — They indicated they can't talk RIGHT NOW but didn't refuse outright:\n"
        "  - Time constraints: busy hoon, abhi nahi, baad mein, meeting mein hoon\n"
        "  - Activity: driving, kaam pe hoon, office mein\n"
        "  - Reschedule request: kal call karo, baad mein karo\n\n"
        "IMPORTANT: When in doubt, classify as YES. Only classify as NO if the "
        "refusal is unmistakable.\n\n"
        f'Customer said: "{user_input}"\n\n'
        "Classification (one word only — YES, NO, or BUSY):"
    )

    try:
        response = await groq_client.chat.completions.create(
            model=CLASSIFIER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=3,
        )
        result = response.choices[0].message.content.strip().upper()
        result = re.sub(r"[^A-Z]", "", result)

        if result in ("YES", "NO", "BUSY"):
            return result
        return "YES"  # Benefit of the doubt
    except Exception as e:
        logger.error(f"Permission classifier error: {e}")
        return "YES"


async def extract_datetime(user_input: str) -> dict:
    """
    Extract scheduling day/time from Hindi/Hinglish/English user input.

    Uses 70B model which natively understands Hindi numbers (paanch=5),
    Hindi day words (kal=tomorrow), and time expressions (subah=morning).
    No manual dictionaries. No regex. No fallback parsing.
    """
    prompt = (
        "You are extracting scheduling information from a customer's spoken response "
        "during a phone call in India.\n\n"
        "CONTEXT: The AI assistant asked the customer when they'd like a callback. "
        "The customer responded with the text below.\n\n"
        "You natively understand:\n"
        "- Hindi days: kal (tomorrow), parso (day after), aaj (today)\n"
        "- Hindi times: subah (morning), dopahar (afternoon), sham/shaam (evening), raat (night)\n"
        "- Hindi numbers with 'baje': ek=1, do=2, teen=3, char=4, paanch=5, "
        "chhah=6, saat=7, aath=8, nau=9, das=10, gyarah=11, barah=12\n"
        "- English days: Monday through Sunday\n"
        "- English times: morning, afternoon, evening, AM, PM\n"
        "- Hinglish combinations: 'kal 5 baje', 'monday morning', 'parso sham ko'\n\n"
        "RULES:\n"
        '1. If the customer mentioned ANY scheduling indicator (a day, time, or both), '
        "extract it and return JSON.\n"
        '   - If only time given (e.g., "5 baje"), assume day is "kal".\n'
        '   - If only day given (e.g., "parso"), set time to "not specified".\n'
        "2. If the response contains NO scheduling information whatsoever — just filler "
        'words like "haan", "ok", "theek hai", "ji", "bataiye", unclear sounds, '
        "or completely unrelated words — return exactly: VAGUE\n\n"
        "OUTPUT FORMAT (strictly one of these two):\n"
        '- Schedule found: {"status":"TimeGiven","day":"<day>","time":"<time>"}\n'
        "- No schedule info: VAGUE\n\n"
        f'Customer said: "{user_input}"\n\n'
        "Output:"
    )

    try:
        response = await groq_client.chat.completions.create(
            model=CLASSIFIER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=60,
        )
        result_text = response.choices[0].message.content.strip()

        # Try to parse JSON from the response
        match = re.search(r"\{.*?\}", result_text)
        if match:
            try:
                data = json.loads(match.group(0))
                if data.get("status") == "TimeGiven":
                    return data
            except json.JSONDecodeError:
                pass

        # Model said VAGUE or output was unparseable
        return {"status": "Vague"}

    except Exception as e:
        logger.error(f"DateTime extraction error: {e}")
        return {"status": "Vague"}


async def classify_bot_type(category: str) -> str:
    """
    Intelligently route a lead to the correct bot based on their category/profession.

    Uses 70B model to understand semantics — e.g. "dentist" → insurance,
    "chartered accountant" → recruitment. No hardcoded keyword dictionaries.
    """
    if not category or not category.strip():
        return "investment"  # Default when no category info

    prompt = (
        "You are routing a sales call to the correct AI bot based on the customer's profession/category.\n\n"
        "We have exactly 3 bots:\n"
        "- INVESTMENT: For anyone who could benefit from financial planning, mutual funds, SIPs, or wealth management. "
        "This includes business owners, professionals, salaried individuals, HNIs, etc.\n"
        "- INSURANCE: For health-related professionals or anyone in the healthcare/medical field "
        "(doctors, clinics, hospitals, dentists, physiotherapists, etc.) — they need health insurance coverage.\n"
        "- RECRUITMENT: For existing financial professionals (insurance agents, mutual fund distributors, "
        "financial advisors, CAs, wealth managers) — we want to recruit them as partners.\n\n"
        "RULE: When in doubt, classify as INVESTMENT (it's our broadest, most versatile bot).\n\n"
        f'Customer category: "{category}"\n\n'
        "Classification (one word only — INVESTMENT, INSURANCE, or RECRUITMENT):"
    )

    try:
        response = await groq_client.chat.completions.create(
            model=CLASSIFIER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=3,
        )
        result = response.choices[0].message.content.strip().upper()
        result = re.sub(r"[^A-Z]", "", result)

        bot_map = {"INVESTMENT": "investment", "INSURANCE": "insurance", "RECRUITMENT": "recruitment"}
        if result in bot_map:
            logger.info(f"LLM classified category '{category}' → {bot_map[result]}")
            return bot_map[result]
        logger.warning(f"LLM returned unexpected bot type '{result}' for category '{category}', defaulting to investment")
        return "investment"
    except Exception as e:
        logger.error(f"Bot type classification error for category '{category}': {e}. Defaulting to investment.")
        return "investment"  # Safe default
