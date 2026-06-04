"""
Kalpvruksh Finserv AI Automation — Voice Call Manager
Handles Bolna AI agent creation, call triggering, call logging, and cost tracking.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx

from server.config import config

logger = logging.getLogger(__name__)

# -------------------------------------------------------
# Voice Profiles
# -------------------------------------------------------
VOICE_PROFILES = {
    "riya": {
        "bot_name": "Riya - Investment Advisor",
        "gender": "female",
        "prompt_file": "investment_bot_prompt.txt",
        "welcome_message": "Hello! Namaste! Kaisi hain aap? Main Riya, Kalpvruksh Finserv se — actually aapko ek interesting cheez batani thi regarding your savings!",
        "synthesizer": {
            "provider": "azuretts",
            "provider_config": {
                "voice": "hi-IN-SwaraNeural",
                "model": "neural",
                "language": "hi-IN",
            },
            "voice_id": "hi-IN-SwaraNeural",
            "stream": True,
            "buffer_size": 100,
            "audio_format": "wav",
        },
    },
    "aarav": {
        "bot_name": "Aarav - Insurance Advisor",
        "gender": "male",
        "prompt_file": "insurance_bot_prompt.txt",
        "welcome_message": (
            "Hi! This is Aarav from Kalpvruksh Finserv, Pune. "
            "Am I speaking with the right person?"
        ),
        "synthesizer": {
            "provider": "polly",
            "provider_config": {
                "voice": "Aditi",
                "engine": "neural",
                "language": "hi-IN"
            },
            "stream": True,
            "buffer_size": 100,
            "audio_format": "wav",
        },
    },
}

# Current active stack info
ACTIVE_STACK = {
    "orchestrator": "Self-hosted (FastAPI + WebSocket)",
    "llm_provider": "Groq",
    "llm_model": "llama-3.3-70b-versatile",
    "stt_provider": "Deepgram",
    "stt_model": "Nova-2 (Hindi)",
    "tts_provider": "AWS Polly",
    "tts_model": "Neural (Kajal hi-IN)",
    "telephony": "Exotel (India SIP)",
}


# -------------------------------------------------------
# Call History Storage (local JSON)
# -------------------------------------------------------
CALL_HISTORY_FILE = config.DATA_DIR / "local_sheets" / "voice_call_logs.json"


def _load_call_history() -> list[dict]:
    """Load call history from local JSON."""
    CALL_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if CALL_HISTORY_FILE.exists():
        try:
            return json.loads(CALL_HISTORY_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, Exception):
            return []
    return []


def _save_call_history(records: list[dict]):
    """Save call history to local JSON."""
    CALL_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    CALL_HISTORY_FILE.write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def log_call_record(record: dict):
    """Append a call record to local storage."""
    records = _load_call_history()
    records.append(record)
    _save_call_history(records)
    logger.info(f"Call logged: {record.get('phone')} → {record.get('bot')}")


def get_call_history() -> list[dict]:
    """Return all call records."""
    return _load_call_history()


def get_cost_summary() -> dict:
    """Calculate cost summary from call history."""
    records = _load_call_history()
    total_calls = len(records)
    total_cost = sum(r.get("estimated_cost_inr", 0) for r in records)
    total_duration = sum(r.get("duration_seconds", 0) for r in records)

    by_bot = {}
    for r in records:
        bot = r.get("bot", "unknown")
        if bot not in by_bot:
            by_bot[bot] = {"calls": 0, "cost": 0, "duration": 0}
        by_bot[bot]["calls"] += 1
        by_bot[bot]["cost"] += r.get("estimated_cost_inr", 0)
        by_bot[bot]["duration"] += r.get("duration_seconds", 0)

    return {
        "total_calls": total_calls,
        "total_cost_inr": round(total_cost, 2),
        "total_duration_seconds": total_duration,
        "avg_cost_per_call_inr": round(total_cost / total_calls, 2) if total_calls else 0,
        "avg_duration_seconds": round(total_duration / total_calls) if total_calls else 0,
        "by_bot": by_bot,
        "cost_breakdown_per_minute": {
            "bolna_telephony": config.COST_BOLNA_TELEPHONY,
            "elevenlabs_tts": config.COST_ELEVENLABS_TTS,
            "deepgram_stt": config.COST_DEEPGRAM_STT,
            "openai_llm": config.COST_OPENAI_LLM,
            "total_per_minute": (
                config.COST_BOLNA_TELEPHONY
                + config.COST_ELEVENLABS_TTS
                + config.COST_DEEPGRAM_STT
                + config.COST_OPENAI_LLM
            ),
        },
    }


def estimate_call_cost(duration_seconds: int, stack: str = "exotel") -> float:
    """Estimate cost of a call based on duration and providers used.
    
    Exotel stack (current production):
    - Telephony: ₹0.80/min (Exotel)
    - STT: ₹0.36/min (Deepgram Nova-2)
    - LLM: ₹0.00/min (Groq free tier)
    - TTS: ₹1.07/min (AWS Polly Neural)
    Total: ~₹2.23/min = ~₹7.80 per 3.5 min call
    """
    minutes = duration_seconds / 60.0
    if stack == "exotel":
        cost = (
            (0.80 * minutes)   # Exotel telephony
            + (0.36 * minutes) # Deepgram STT
            + (0.00)           # Groq LLM (free)
            + (1.07 * minutes) # AWS Polly TTS
        )
    else:
        # Legacy Bolna stack
        tts_cost = config.COST_ELEVENLABS_TTS
        cost = (
            (config.COST_BOLNA_TELEPHONY * minutes)
            + (tts_cost * minutes)
            + (config.COST_DEEPGRAM_STT * minutes)
            + config.COST_OPENAI_LLM
        )
    return round(cost, 2)


# -------------------------------------------------------
# Bolna API Integration
# -------------------------------------------------------
async def create_agent_and_call(
    phone: str,
    bot_type: str = "riya",
) -> dict:
    """
    Create a Bolna agent and make a call.

    Args:
        phone: Phone number with country code (e.g. +919022873952)
        bot_type: 'riya' (investment) or 'aarav' (insurance)

    Returns:
        dict with agent_id, call_id, status
    """
    if not config.BOLNA_API_KEY:
        return {"status": "error", "message": "BOLNA_API_KEY not configured"}

    profile = VOICE_PROFILES.get(bot_type)
    if not profile:
        return {"status": "error", "message": f"Unknown bot_type: {bot_type}"}

    # Ensure phone has + prefix
    if not phone.startswith("+"):
        phone = f"+91{phone}" if not phone.startswith("91") else f"+{phone}"

    # Load prompt
    prompt_path = config.PROMPTS_DIR / profile["prompt_file"]
    try:
        system_prompt = config.load_prompt(prompt_path)
    except FileNotFoundError:
        system_prompt = f"You are {bot_type.title()}, a financial advisor at Kalpvruksh Finserv, Pune."

    # Build agent payload
    agent_payload = {
        "agent_config": {
            "agent_name": profile["bot_name"],
            "agent_type": "other",
            "agent_welcome_message": profile["welcome_message"],
            "tasks": [
                {
                    "task_type": "conversation",
                    "toolchain": {
                        "execution": "parallel",
                        "pipelines": [["transcriber", "llm", "synthesizer"]],
                    },
                    "tools_config": {
                        "llm_agent": {
                            "agent_type": "simple_llm_agent",
                            "agent_flow_type": "streaming",
                            "llm_config": {
                                "provider": "openai",
                                "family": "openai",
                                "model": "gpt-4o-mini",
                                "max_tokens": 80,
                                "temperature": 0.7,
                                "agent_flow_type": "streaming",
                            },
                        },
                        "transcriber": {
                            "provider": "deepgram",
                            "model": "nova-2",
                            "language": "hi",
                            "stream": True,
                            "endpointing": 200,
                            "keywords": "SIP, mutual fund, insurance, FD, Kalpvruksh, investment, SIP returns",
                        },
                        "synthesizer": profile["synthesizer"],
                        "input": {"provider": "twilio", "format": "wav"},
                        "output": {"provider": "twilio", "format": "wav"},
                    },
                    "task_config": {
                        "hangup_after_silence": 15,
                        "incremental_delay": 300,
                        "number_of_words_for_interruption": 2,
                        "call_terminate": 240,
                        "backchanneling": True,
                        "backchanneling_message_gap": 4,
                        "backchanneling_start_delay": 3,
                        "ambient_noise_track": "office-ambience",
                        "optimize_latency": True,
                    },
                }
            ],
        },
        "agent_prompts": {"task_1": {"system_prompt": system_prompt}},
    }

    headers = {
        "Authorization": f"Bearer {config.BOLNA_API_KEY}",
        "Content-Type": "application/json",
    }

    result = {
        "phone": phone,
        "bot": profile["bot_name"],
        "bot_type": bot_type,
        "timestamp": datetime.now().isoformat(),
        "status": "pending",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Step 1: Create agent
        try:
            resp = await client.post(
                f"{config.BOLNA_BASE_URL}/v2/agent",
                json=agent_payload,
                headers=headers,
            )
            if resp.status_code in (200, 201):
                agent_data = resp.json()
                result["agent_id"] = agent_data.get("agent_id", agent_data.get("id", ""))
                logger.info(f"Agent created: {result['agent_id']}")
            else:
                result["status"] = "error"
                result["message"] = f"Agent creation failed: {resp.status_code} — {resp.text}"
                logger.error(result["message"])
                return result
        except Exception as e:
            result["status"] = "error"
            result["message"] = f"Agent creation error: {str(e)}"
            logger.error(result["message"])
            return result

        # Step 2: Make the call
        try:
            resp = await client.post(
                f"{config.BOLNA_BASE_URL}/call",
                json={
                    "agent_id": result["agent_id"],
                    "recipient_phone_number": phone,
                },
                headers=headers,
            )
            if resp.status_code in (200, 201):
                call_data = resp.json()
                result["call_id"] = call_data.get("call_id", call_data.get("id", ""))
                result["status"] = "initiated"
                logger.info(f"Call initiated to {phone}")
            else:
                result["status"] = "error"
                result["message"] = f"Call failed: {resp.status_code} — {resp.text}"
                logger.error(result["message"])
        except Exception as e:
            result["status"] = "error"
            result["message"] = f"Call error: {str(e)}"
            logger.error(result["message"])

    # Log the call attempt
    log_call_record(result)
    return result


def process_bolna_webhook(payload: dict) -> dict:
    """
    Process a Bolna call completion webhook.
    Extracts lead data, scores it, and logs to call history.

    Returns the processed record.
    """
    # Extract fields from Bolna webhook payload
    agent_id = payload.get("agent_id", "")
    call_id = payload.get("call_id", "")
    status = payload.get("status", "completed")
    duration = payload.get("duration", 0)
    transcript = payload.get("transcript", "")
    phone = payload.get("recipient_phone_number", payload.get("phone", ""))
    summary = payload.get("summary", "")

    # Determine bot type from agent name or context
    agent_name = payload.get("agent_name", "")
    if "riya" in agent_name.lower() or "investment" in agent_name.lower():
        bot_type = "riya"
        bot_label = "Riya (Investment)"
    else:
        bot_type = "aarav"
        bot_label = "Aarav (Insurance)"

    # Estimate cost
    estimated_cost = estimate_call_cost(duration)

    # Determine interest level from transcript/summary
    interest = "unknown"
    lead_score = 3  # default cold
    lower_text = (transcript + " " + summary).lower()

    if any(w in lower_text for w in ["interested", "yes", "sure", "tell me more", "haan", "batao"]):
        interest = "interested"
        lead_score = 8
    elif any(w in lower_text for w in ["maybe", "sochta", "sochke", "think about"]):
        interest = "maybe"
        lead_score = 5
    elif any(w in lower_text for w in ["not interested", "no", "nahi", "busy", "don't call"]):
        interest = "not_interested"
        lead_score = 1

    # Determine lead status
    if lead_score >= 7:
        lead_status = "HOT"
    elif lead_score >= 4:
        lead_status = "WARM"
    else:
        lead_status = "COLD"

    record = {
        "timestamp": datetime.now().isoformat(),
        "phone": phone,
        "bot": bot_label,
        "bot_type": bot_type,
        "agent_id": agent_id,
        "call_id": call_id,
        "duration_seconds": duration,
        "estimated_cost_inr": estimated_cost,
        "lead_score": lead_score,
        "lead_status": lead_status,
        "interest_level": interest,
        "customer_name": payload.get("customer_name", ""),
        "conversation_summary": summary or transcript[:500],
        "manager_action": "PENDING" if lead_status == "HOT" else "NONE",
        "call_status": status,
    }

    log_call_record(record)
    logger.info(
        f"Webhook processed: {phone} → {bot_label} | "
        f"Score: {lead_score} ({lead_status}) | Cost: ₹{estimated_cost}"
    )

    return record


def get_system_status() -> dict:
    """Return full system status with all APIs, models, and providers."""
    return {
        "service": "Kalpvruksh Finserv AI Automation",
        "version": "1.1.0",
        "status": "active",
        "bots": {
            "riya": {
                "role": "Investment Advisor",
                "gender": "Female",
                "voice": "ElevenLabs Apsara (Natural Hinglish)",
                "prompt": "investment_bot_prompt.txt",
            },
            "aarav": {
                "role": "Insurance Advisor",
                "gender": "Male",
                "voice": "ElevenLabs Arnav (Calm & Friendly)",
                "prompt": "insurance_bot_prompt.txt",
            },
        },
        "active_stack": ACTIVE_STACK,
        "api_keys_status": {
            "bolna_ai": "✅ Connected" if config.BOLNA_API_KEY else "❌ Missing",
            "sarvam_ai": "✅ Connected" if config.SARVAM_API_KEY else "❌ Missing",
            "groq": "✅ Connected" if config.GROQ_API_KEY else "❌ Missing",
            "openai": "✅ Connected" if config.OPENAI_API_KEY else "⚠️ Not set (using Bolna's built-in)",
            "google_sheets": "✅ Connected" if config.LEADS_SHEET_ID else "⚠️ Using local JSON fallback",
        },
        "cost_per_minute_inr": {
            "telephony (Bolna)": config.COST_BOLNA_TELEPHONY,
            "voice (ElevenLabs)": config.COST_ELEVENLABS_TTS,
            "ears (Deepgram)": config.COST_DEEPGRAM_STT,
            "brain (GPT-4o-mini)": config.COST_OPENAI_LLM,
            "total": (
                config.COST_BOLNA_TELEPHONY
                + config.COST_ELEVENLABS_TTS
                + config.COST_DEEPGRAM_STT
                + config.COST_OPENAI_LLM
            ),
        },
        "estimated_cost_per_call_inr": {
            "2_min_call": estimate_call_cost(120),
            "3_min_call": estimate_call_cost(180),
            "5_min_call": estimate_call_cost(300),
        },
        "call_stats": get_cost_summary(),
    }
