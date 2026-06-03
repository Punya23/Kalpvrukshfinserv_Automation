"""
Kalpvruksh Finserv AI Automation — Main Server
FastAPI application with webhook routes for WhatsApp, Voice AI, and testing.
"""

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from server.config import config
from server.orchestrator import classify_intent
from server.bots.insurance_bot import insurance_bot
from server.bots.investment_bot import investment_bot
from server.bots.reminder_bot import reminder_bot
from server.scheduler import renewal_scheduler
from server.lead_scoring import LeadSource

# -------------------------------------------------------
# Logging Setup
# -------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG if config.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("kalpvruksh")


# -------------------------------------------------------
# Scheduler Setup (runs renewal checks daily at 9 AM)
# -------------------------------------------------------
scheduler = AsyncIOScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    # Startup
    logger.info("=" * 60)
    logger.info("🌳 KALPVRUKSH FINSERV AI AUTOMATION — Starting...")
    logger.info("=" * 60)

    # Validate configuration
    warnings = config.validate()
    for w in warnings:
        logger.warning(f"⚠️  {w}")

    # Start the renewal scheduler (checks daily at 9:00 AM IST)
    scheduler.add_job(
        renewal_scheduler.check_and_send_reminders,
        "cron",
        hour=9,
        minute=0,
        id="daily_renewal_check",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("📅 Renewal scheduler started (daily at 09:00 AM)")
    logger.info(f"🤖 LLM Provider: {config.LLM_PROVIDER} ({config.LLM_MODEL})")
    logger.info(f"🌐 Server: http://{config.SERVER_HOST}:{config.SERVER_PORT}")
    logger.info("=" * 60)

    yield

    # Shutdown
    scheduler.shutdown()
    logger.info("🌳 Kalpvruksh Finserv AI Automation — Stopped.")


# -------------------------------------------------------
# FastAPI App
# -------------------------------------------------------
app = FastAPI(
    title="Kalpvruksh Finserv AI Automation",
    description="Voice & Text Bot Automation for Insurance, Investment, and Reminder services.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware (allow all for testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------
# Request/Response Models
# -------------------------------------------------------
class ChatRequest(BaseModel):
    """Request model for the chat endpoint."""
    message: str
    session_id: Optional[str] = None
    phone: Optional[str] = None
    customer_id: Optional[str] = None
    source: Optional[str] = "inbound_whatsapp"
    # If bot_type is specified, skip orchestrator and route directly
    bot_type: Optional[str] = None  # "insurance", "investment", "reminder"


class ChatResponse(BaseModel):
    """Response model for the chat endpoint."""
    response: str
    session_id: str
    intent: Optional[str] = None
    confidence: Optional[float] = None
    bot_used: str


class RenewalSummaryResponse(BaseModel):
    """Response model for the renewal dashboard."""
    total_upcoming: int
    due_7_days: list
    due_30_days: list
    due_60_days: list
    overdue: list
    total_premium_at_risk: int


# -------------------------------------------------------
# Routes
# -------------------------------------------------------

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "active",
        "service": "Kalpvruksh Finserv AI Automation",
        "version": "1.0.0",
        "bots": ["insurance (Aarav)", "investment (Riya)", "reminder (Vikram)"],
        "llm_provider": config.LLM_PROVIDER,
        "model": config.LLM_MODEL,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint — handles all incoming messages.

    Flow:
    1. If bot_type is specified → route directly to that bot
    2. Otherwise → Orchestrator classifies intent → routes to correct bot
    3. Bot processes message → returns response
    4. If hot lead detected → logs to Google Sheet + alerts manager
    """
    session_id = request.session_id or str(uuid.uuid4())

    try:
        source = LeadSource(request.source) if request.source else LeadSource.INBOUND_WHATSAPP
    except ValueError:
        source = LeadSource.INBOUND_WHATSAPP

    # Determine which bot to use
    if request.bot_type:
        intent = request.bot_type.upper()
        confidence = 1.0
    else:
        # Use orchestrator to classify intent
        intent_result = classify_intent(request.message)
        intent = intent_result.intent
        confidence = intent_result.confidence
        logger.info(f"Orchestrator: {intent} (confidence: {confidence:.2f}) — {intent_result.reason}")

    # Route to the correct bot
    if intent == "INSURANCE":
        response_text = await insurance_bot.handle_message(session_id, request.message, source)
        bot_used = "insurance (Aarav)"
    elif intent == "INVESTMENT":
        response_text = await investment_bot.handle_message(session_id, request.message, source)
        bot_used = "investment (Riya)"
    elif intent == "REMINDER":
        response_text = await reminder_bot.handle_message(
            session_id, request.message,
            customer_id=request.customer_id,
            phone=request.phone,
        )
        bot_used = "reminder (Vikram)"
    else:
        # UNKNOWN intent — ask for clarification
        response_text = (
            "Namaste! Kalpvruksh Finserv mein aapka swagat hai. 🌳\n\n"
            "Main aapki kaise madad kar sakta hoon?\n\n"
            "1️⃣ Health/Life Insurance ke baare mein jaanein\n"
            "2️⃣ Investment/SIP/Mutual Fund ke baare mein jaanein\n"
            "3️⃣ Apni existing policy ya investment ka status check karein\n\n"
            "Koi bhi option choose karein ya apna sawaal seedha poochein!"
        )
        bot_used = "orchestrator"

    return ChatResponse(
        response=response_text,
        session_id=session_id,
        intent=intent,
        confidence=confidence,
        bot_used=bot_used,
    )


@app.post("/webhook/whatsapp")
async def whatsapp_webhook(request: Request):
    """
    WhatsApp webhook endpoint.
    Receives incoming messages from WhatsApp Business API (Interakt/Wati/Meta).
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # Extract message (format varies by WhatsApp API provider)
    # This handles the common Interakt/Meta format
    message_text = ""
    phone_number = ""

    # Try Meta/Interakt format
    if "entry" in body:
        # Meta Cloud API format
        try:
            changes = body["entry"][0]["changes"][0]["value"]
            if "messages" in changes:
                msg = changes["messages"][0]
                message_text = msg.get("text", {}).get("body", "")
                phone_number = msg.get("from", "")
        except (KeyError, IndexError):
            pass
    elif "message" in body:
        # Simple format
        message_text = body.get("message", "")
        phone_number = body.get("phone", body.get("from", ""))
    elif "text" in body:
        message_text = body.get("text", "")
        phone_number = body.get("phone", "")

    if not message_text:
        return JSONResponse({"status": "no_message"}, status_code=200)

    # Process through the main chat handler
    chat_request = ChatRequest(
        message=message_text,
        phone=phone_number,
        session_id=f"wa-{phone_number}",
        source="inbound_whatsapp",
    )

    result = await chat(chat_request)

    # In production, send the response back via WhatsApp API
    # For now, log it
    logger.info(f"WhatsApp [{phone_number}] → [{result.bot_used}]: {result.response[:100]}...")

    return JSONResponse({
        "status": "processed",
        "bot_used": result.bot_used,
        "response_preview": result.response[:200],
    })


@app.post("/webhook/vapi")
async def vapi_webhook(request: Request):
    """
    Vapi.ai voice webhook endpoint.
    Receives voice conversation events from Vapi.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    event_type = body.get("message", {}).get("type", body.get("type", ""))

    if event_type == "function-call":
        # Handle Vapi function calls
        function_name = body.get("message", {}).get("functionCall", {}).get("name", "")
        parameters = body.get("message", {}).get("functionCall", {}).get("parameters", {})
        logger.info(f"Vapi function call: {function_name}({parameters})")

        # Route to appropriate handler
        return JSONResponse({"result": "Function processed"})

    elif event_type == "transcript":
        # Process transcript
        transcript = body.get("message", {}).get("transcript", "")
        logger.info(f"Vapi transcript: {transcript[:100]}...")
        return JSONResponse({"status": "received"})

    elif event_type == "end-of-call-report":
        logger.info("Vapi call ended")
        return JSONResponse({"status": "noted"})

    return JSONResponse({"status": "ok"})


@app.get("/renewals/summary")
async def renewal_summary():
    """Get a summary of upcoming renewals for the dashboard."""
    summary = await renewal_scheduler.get_renewal_summary()
    return summary


@app.post("/renewals/trigger-check")
async def trigger_renewal_check():
    """Manually trigger a renewal check (for testing)."""
    await renewal_scheduler.check_and_send_reminders()
    return {"status": "Renewal check completed"}


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation history for a session."""
    insurance_bot.clear_session(session_id)
    investment_bot.clear_session(session_id)
    reminder_bot.clear_session(session_id)
    return {"status": f"Session {session_id} cleared"}


@app.get("/health")
async def health_check():
    """Detailed health check with all component statuses."""
    return {
        "status": "healthy",
        "components": {
            "llm": {
                "provider": config.LLM_PROVIDER,
                "model": config.LLM_MODEL,
                "api_key_set": bool(config.GROQ_API_KEY or config.OPENAI_API_KEY),
            },
            "google_sheets": {
                "sheet_id_set": bool(config.LEADS_SHEET_ID),
                "credentials_exists": config.GOOGLE_SHEETS_CREDENTIALS_FILE != "",
            },
            "whatsapp": {
                "api_key_set": bool(config.WHATSAPP_API_KEY),
                "manager_number_set": bool(config.MANAGER_WHATSAPP_NUMBER),
            },
            "scheduler": {
                "running": scheduler.running,
                "next_run": str(scheduler.get_jobs()[0].next_run_time) if scheduler.get_jobs() else "No jobs",
            },
        },
    }


# -------------------------------------------------------
# Entry Point
# -------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server.main:app",
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        reload=config.DEBUG,
    )
