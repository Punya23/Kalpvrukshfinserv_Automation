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
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "cerebras")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.3-70b")

    # --- Cerebras (primary — fastest free inference, 30k tokens/min) ---
    # Free models: gpt-oss-120b (best quality), gemma-4-31b (fast fallback)
    CEREBRAS_API_KEY: str = os.getenv("CEREBRAS_API_KEY", "")
    CEREBRAS_MODEL: str = os.getenv("CEREBRAS_MODEL", "gpt-oss-120b")
    CEREBRAS_MODEL_FALLBACK: str = os.getenv("CEREBRAS_MODEL_FALLBACK", "gemma-4-31b")
    CEREBRAS_BASE_URL: str = "https://api.cerebras.ai/v1"

    # --- OpenRouter (fallback — no daily quota, routes to fastest provider) ---
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")
    OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    # --- Groq (last resort fallback) ---
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # --- Phase 3 Voice Pipeline (Self-Hosted) ---
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    DEEPGRAM_API_KEY: str = os.getenv("DEEPGRAM_API_KEY", "")

    # --- Exotel (India Telephony) ---
    EXOTEL_API_KEY: str = os.getenv("EXOTEL_API_KEY", "")
    EXOTEL_API_TOKEN: str = os.getenv("EXOTEL_API_TOKEN", "")
    EXOTEL_ACCOUNT_SID: str = os.getenv("EXOTEL_ACCOUNT_SID", "")
    EXOTEL_CALLER_ID: str = os.getenv("EXOTEL_CALLER_ID", "")
    EXOTEL_APP_ID: str = os.getenv("EXOTEL_APP_ID", "")
    EXOTEL_SUBDOMAIN: str = os.getenv("EXOTEL_SUBDOMAIN", "api.exotel.com")

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

    # --- Campaign ---
    # Max times a single number is dialed across ALL campaigns before it is
    # permanently skipped (guards against re-calling a no-answer number forever).
    MAX_CALL_ATTEMPTS: int = int(os.getenv("MAX_CALL_ATTEMPTS", "3"))

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
        if not cls.CEREBRAS_API_KEY and not cls.OPENROUTER_API_KEY and not cls.GROQ_API_KEY:
            warnings.append("No LLM API key set (CEREBRAS_API_KEY / OPENROUTER_API_KEY / GROQ_API_KEY) — LLM calls will fail")
        if not cls.DEEPGRAM_API_KEY:
            warnings.append("DEEPGRAM_API_KEY not set — STT (speech-to-text) disabled")
        if not cls.EXOTEL_API_KEY:
            warnings.append("EXOTEL_API_KEY not set — outbound calls will fail")
        if not cls.LEADS_SHEET_ID:
            warnings.append("LEADS_SHEET_ID not set — Google Sheets logging disabled")
        if not cls.WHATSAPP_API_KEY:
            warnings.append("WHATSAPP_API_KEY not set — WhatsApp notifications disabled")
        return warnings


# Singleton instance
config = Config()
