"""
Kalpvruksh Finserv AI Automation — Configuration Module
Handles all environment variables and configuration loading.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)


class Config:
    """Central configuration for the AI automation system."""

    # --- LLM Provider ---
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")  # "groq" (free) or "openai" (paid)
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

    # --- Voice AI (Vapi.ai) ---
    VAPI_API_KEY: str = os.getenv("VAPI_API_KEY", "")
    VAPI_PHONE_NUMBER_ID: str = os.getenv("VAPI_PHONE_NUMBER_ID", "")

    # --- Bolna AI (Voice Calls) ---
    BOLNA_API_KEY: str = os.getenv("BOLNA_API_KEY", "")
    BOLNA_BASE_URL: str = "https://api.bolna.ai"

    # --- Sarvam AI (Hindi Voice) ---
    SARVAM_API_KEY: str = os.getenv("SARVAM_API_KEY", "")

    # --- Phase 3 Voice Pipeline (Self-Hosted) ---
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    TWILIO_ACCOUNT_SID: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    TWILIO_AUTH_TOKEN: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    TWILIO_PHONE_NUMBER: str = os.getenv("TWILIO_PHONE_NUMBER", "")
    DEEPGRAM_API_KEY: str = os.getenv("DEEPGRAM_API_KEY", "")

    # --- Exotel (India Telephony) ---
    EXOTEL_API_KEY: str = os.getenv("EXOTEL_API_KEY", "")
    EXOTEL_API_TOKEN: str = os.getenv("EXOTEL_API_TOKEN", "")
    EXOTEL_ACCOUNT_SID: str = os.getenv("EXOTEL_ACCOUNT_SID", "")
    EXOTEL_CALLER_ID: str = os.getenv("EXOTEL_CALLER_ID", "")
    EXOTEL_APP_ID: str = os.getenv("EXOTEL_APP_ID", "")
    EXOTEL_SUBDOMAIN: str = os.getenv("EXOTEL_SUBDOMAIN", "api.exotel.com")

    # --- Voice Call Cost Estimates (INR per minute) ---
    COST_BOLNA_TELEPHONY: float = 2.0    # Bolna platform + Twilio telephony
    COST_ELEVENLABS_TTS: float = 8.0     # ElevenLabs Turbo v2.5
    COST_SARVAM_TTS: float = 1.5         # Sarvam Bulbul v3
    COST_DEEPGRAM_STT: float = 0.5       # Deepgram Nova-2
    COST_OPENAI_LLM: float = 1.5         # GPT-4o-mini per call (flat estimate)
    COST_GROQ_LLM: float = 0.0           # Groq is free

    # --- Google Sheets ---
    GOOGLE_SHEETS_CREDENTIALS_FILE: str = os.getenv(
        "GOOGLE_SHEETS_CREDENTIALS_FILE", "credentials/google_service_account.json"
    )
    LEADS_SHEET_ID: str = os.getenv("LEADS_SHEET_ID", "")
    LEADS_SHEET_NAME: str = os.getenv("LEADS_SHEET_NAME", "Hot Leads")
    NURTURE_SHEET_NAME: str = os.getenv("NURTURE_SHEET_NAME", "Nurture Pipeline")
    RENEWALS_SHEET_NAME: str = os.getenv("RENEWALS_SHEET_NAME", "Renewals Tracker")

    # --- WhatsApp ---
    WHATSAPP_API_URL: str = os.getenv("WHATSAPP_API_URL", "")
    WHATSAPP_API_KEY: str = os.getenv("WHATSAPP_API_KEY", "")
    MANAGER_WHATSAPP_NUMBER: str = os.getenv("MANAGER_WHATSAPP_NUMBER", "")

    # --- Manager ---
    MANAGER_NAME: str = os.getenv("MANAGER_NAME", "Sanjeev Surana")
    MANAGER_PHONE: str = os.getenv("MANAGER_PHONE", "")
    MANAGER_EMAIL: str = os.getenv("MANAGER_EMAIL", "")

    # --- Server ---
    SERVER_HOST: str = os.getenv("SERVER_HOST", "0.0.0.0")
    SERVER_PORT: int = int(os.getenv("SERVER_PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"

    # --- Prompt Files ---
    PROMPTS_DIR: Path = PROJECT_ROOT / "prompts"
    ORCHESTRATOR_PROMPT: Path = PROMPTS_DIR / "orchestrator_prompt.txt"
    INSURANCE_BOT_PROMPT: Path = PROMPTS_DIR / "insurance_bot_prompt.txt"
    INVESTMENT_BOT_PROMPT: Path = PROMPTS_DIR / "investment_bot_prompt.txt"
    REMINDER_BOT_PROMPT: Path = PROMPTS_DIR / "reminder_bot_prompt.txt"

    # --- Data ---
    DATA_DIR: Path = PROJECT_ROOT / "data"

    @classmethod
    def load_prompt(cls, prompt_path: Path) -> str:
        """Load a prompt file and return its contents as a string."""
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        return prompt_path.read_text(encoding="utf-8")

    @classmethod
    def validate(cls) -> list[str]:
        """Validate that required configuration is present. Returns list of warnings."""
        warnings = []
        if cls.LLM_PROVIDER == "groq" and not cls.GROQ_API_KEY:
            warnings.append("GROQ_API_KEY is not set — LLM calls will fail")
        if cls.LLM_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            warnings.append("OPENAI_API_KEY is not set — LLM calls will fail")
        if not cls.LEADS_SHEET_ID:
            warnings.append("LEADS_SHEET_ID not set — Google Sheets logging disabled")
        if not cls.WHATSAPP_API_KEY:
            warnings.append("WHATSAPP_API_KEY not set — WhatsApp notifications disabled")
        return warnings


# Singleton instance
config = Config()
