"""
Kalpvruksh Finserv AI Automation — Orchestrator
Central intent classifier that routes incoming messages to the correct bot.
"""

import json
import logging
from typing import Optional

from server.config import config

logger = logging.getLogger(__name__)

# -------------------------------------------------------
# LLM Client Setup — Groq primary, OpenRouter fallback (mirrors the voice
# pipeline's provider order in server/llm_client.py). This sync client is only
# for the /chat text endpoint; the voice pipeline uses the shared async client.
# We also pin the matching model id — config.LLM_MODEL is a Cerebras model
# (gpt-oss-120b) that does NOT exist on Groq/OpenRouter and would 404.
# -------------------------------------------------------

from openai import OpenAI as _OpenAI

_llm_client = None
_llm_model = None
if config.GROQ_API_KEY:
    from groq import Groq as _Groq
    _llm_client = _Groq(api_key=config.GROQ_API_KEY)
    _llm_model = config.GROQ_MODEL
elif config.OPENROUTER_API_KEY:
    _llm_client = _OpenAI(
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL,
    )
    _llm_model = config.OPENROUTER_MODEL
else:
    logger.warning("No LLM client configured — /chat will use keyword fallback")


# Load orchestrator system prompt
try:
    ORCHESTRATOR_SYSTEM_PROMPT = config.load_prompt(config.ORCHESTRATOR_PROMPT)
except FileNotFoundError:
    ORCHESTRATOR_SYSTEM_PROMPT = "Classify the user message as INSURANCE, INVESTMENT, REMINDER, or UNKNOWN."
    logger.warning("Orchestrator prompt file not found, using fallback.")


class IntentResult:
    """Result of intent classification."""

    def __init__(self, intent: str, confidence: float, reason: str,
                 language: str, is_existing_customer: bool,
                 customer_id: Optional[str] = None):
        self.intent = intent  # INSURANCE, INVESTMENT, REMINDER, UNKNOWN
        self.confidence = confidence
        self.reason = reason
        self.language = language
        self.is_existing_customer = is_existing_customer
        self.customer_id = customer_id

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            "reason": self.reason,
            "language_detected": self.language,
            "is_existing_customer": self.is_existing_customer,
            "customer_id_found": self.customer_id,
        }


def classify_intent(user_message: str) -> IntentResult:
    """
    Use the LLM to classify the user's intent and route to the correct bot.
    Falls back to keyword-based classification if LLM is unavailable.
    """
    if _llm_client is None:
        logger.warning("LLM client not available, using keyword-based classification.")
        return _keyword_classify(user_message)

    try:
        response = _llm_client.chat.completions.create(
            model=_llm_model,
            messages=[
                {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,  # Low temperature for consistent classification
            max_tokens=300,
        )

        # Guard against None content (reasoning-token exhaustion returns 200 + None)
        response_text = (response.choices[0].message.content or "").strip()
        if not response_text:
            logger.warning("Orchestrator LLM returned empty content — keyword fallback.")
            return _keyword_classify(user_message)

        # Parse JSON response
        # Handle cases where LLM wraps JSON in markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        parsed = json.loads(response_text)

        return IntentResult(
            intent=parsed.get("intent", "UNKNOWN"),
            confidence=parsed.get("confidence", 0.5),
            reason=parsed.get("reason", ""),
            language=parsed.get("language_detected", "hinglish"),
            is_existing_customer=parsed.get("is_existing_customer", False),
            customer_id=parsed.get("customer_id_found"),
        )

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse orchestrator response as JSON: {e}")
        return _keyword_classify(user_message)
    except Exception as e:
        logger.error(f"Orchestrator LLM call failed: {e}")
        return _keyword_classify(user_message)


def _keyword_classify(message: str) -> IntentResult:
    """Fallback keyword-based intent classification when LLM is unavailable."""
    msg_lower = message.lower()

    # Check for customer ID pattern first
    import re
    customer_id_match = re.search(r'kf-\d{3,4}', msg_lower)
    if customer_id_match:
        return IntentResult(
            intent="REMINDER",
            confidence=0.9,
            reason="Customer ID detected in message",
            language="unknown",
            is_existing_customer=True,
            customer_id=customer_id_match.group().upper(),
        )

    # Insurance keywords
    insurance_keywords = [
        "insurance", "bima", "policy", "health cover", "mediclaim",
        "star health", "term plan", "premium", "claim", "cashless",
        "hospital", "tpa", "health insurance", "life insurance",
        "suraksha", "beema", "insure",
    ]

    # Investment keywords
    investment_keywords = [
        "invest", "mutual fund", "sip", "returns", "portfolio",
        "tax saving", "elss", "80c", "retirement", "wealth",
        "paisa", "nivesh", "bachat", "savings", "fd", "gold",
        "share", "nifty", "sensex", "fund",
    ]

    # Reminder / Status keywords
    reminder_keywords = [
        "renewal", "renew", "expiry", "status", "meri policy",
        "mera investment", "kitna hua", "portfolio status",
        "account", "statement", "update", "reminder",
    ]

    insurance_score = sum(1 for kw in insurance_keywords if kw in msg_lower)
    investment_score = sum(1 for kw in investment_keywords if kw in msg_lower)
    reminder_score = sum(1 for kw in reminder_keywords if kw in msg_lower)

    max_score = max(insurance_score, investment_score, reminder_score)

    if max_score == 0:
        return IntentResult(
            intent="UNKNOWN",
            confidence=0.3,
            reason="No matching keywords found",
            language="unknown",
            is_existing_customer=False,
        )

    if insurance_score == max_score:
        intent = "INSURANCE"
    elif investment_score == max_score:
        intent = "INVESTMENT"
    else:
        intent = "REMINDER"

    return IntentResult(
        intent=intent,
        confidence=min(0.5 + (max_score * 0.1), 0.9),
        reason=f"Keyword match (score: {max_score})",
        language="unknown",
        is_existing_customer=(intent == "REMINDER"),
    )
