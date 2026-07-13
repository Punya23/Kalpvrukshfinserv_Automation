"""
Kalpvruksh Finserv AI Automation — Main Server
FastAPI application with webhook routes for WhatsApp, Voice AI, and testing.
"""

import json
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
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
from server.voice_pipeline import ExotelVoiceConnectionManager
from server.campaign.campaign_runner import campaign_runner
from server.campaign.trai_compliance import get_calling_status
import requests

# -------------------------------------------------------
# Logging Setup
# -------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG if config.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("kalpvruksh")

# Suppress noisy library logs
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("websockets.client").setLevel(logging.WARNING)


# -------------------------------------------------------
# Scheduler Setup (runs cron jobs in IST — Railway is UTC)
# -------------------------------------------------------
import pytz
_IST = pytz.timezone("Asia/Kolkata")
scheduler = AsyncIOScheduler(timezone=_IST)


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

    # Ensure all data directories exist (Railway ephemeral FS)
    import os as _os
    for _d in ["data/leads", "data/call_logs", "data/campaigns", "data/qa"]:
        _os.makedirs(_d, exist_ok=True)
    logger.info("📁 Data directories ready")

    # Renewal reminders — daily 9:00 AM IST
    scheduler.add_job(
        renewal_scheduler.check_and_send_reminders,
        "cron",
        hour=9,
        minute=0,
        id="daily_renewal_check",
        replace_existing=True,
    )

    # Nightly lead scraper — daily 8:00 PM IST
    scheduler.add_job(
        run_nightly_scrape,
        "cron",
        hour=20,
        minute=0,
        id="nightly_lead_scrape",
        replace_existing=True,
    )

    # Morning auto-campaign — daily 10:00 AM IST (first TRAI optimal window)
    # Only fires on weekdays (Mon-Fri). Uses TELEPHONY_PROVIDER env var.
    scheduler.add_job(
        auto_morning_campaign,
        "cron",
        hour=10,
        minute=0,
        day_of_week="mon-fri",
        id="morning_auto_campaign",
        replace_existing=True,
    )

    scheduler.start()
    logger.info("📅 Renewal scheduler started (daily at 09:00 AM)")
    logger.info("🔍 Lead scraper scheduled (daily at 08:00 PM)")
    logger.info("📞 Morning campaign scheduled (Mon-Fri at 10:00 AM)")
    logger.info(f"🤖 LLM Provider: {config.LLM_PROVIDER} ({config.LLM_MODEL})")
    logger.info(f"🌐 Server: http://{config.SERVER_HOST}:{config.SERVER_PORT}")
    logger.info("=" * 60)

    yield

    # Shutdown
    scheduler.shutdown()
    logger.info("🌳 Kalpvruksh Finserv AI Automation — Stopped.")


async def run_nightly_scrape():
    """
    Nightly lead scraper — runs at 8 PM IST.
    Uses the lead_pipeline collectors to scrape fresh leads,
    deduplicates against existing data, and saves to unified CSV.
    """
    import asyncio
    from server.lead_pipeline.core.csv_manager import UnifiedCSVManager
    from server.lead_pipeline.core.compliance import ComplianceGate

    logger.info("🔍 Nightly lead scrape starting...")

    queries = [
        ("doctors in pune", "doctor"),
        ("architects in pune", "architect"),
        ("CA chartered accountant in pune", "CA"),
        ("interior designers in pune", "interior_designer"),
        ("dentists in pune", "dentist"),
    ]

    csv_manager = UnifiedCSVManager()
    compliance = ComplianceGate()
    total_new = 0

    for query, category in queries:
        try:
            # Use the DDG/Nominatim collector (no browser needed, works on Railway)
            from server.lead_pipeline.collectors.collector_nominatim import NominatimCollector
            collector = NominatimCollector()
            leads = collector.fetch_real_leads(query, limit=15)

            compliant = [l for l in leads if compliance.is_lead_callable(l)]
            if compliant:
                csv_manager.save_leads(compliant)
                total_new += len(compliant)
                logger.info(f"🔍 [{category}] Scraped {len(compliant)} compliant leads")
            else:
                logger.info(f"🔍 [{category}] No new compliant leads")

        except Exception as e:
            logger.error(f"🔍 [{category}] Scrape failed: {e}")
            continue

    logger.info(f"✅ Nightly scrape complete. {total_new} new leads added.")


async def auto_morning_campaign():
    """
    Morning auto-campaign — fires at 10:00 AM IST, Mon-Fri.
    Reads leads from the previous night's scrape and calls them all
    using whichever telephony provider is set in TELEPHONY_PROVIDER env var.
    """
    from server.campaign.campaign_runner import campaign_runner
    from server.campaign.trai_compliance import is_good_calling_day

    if not is_good_calling_day():
        logger.info("[AutoCampaign] Weekend detected — skipping morning campaign.")
        return

    if campaign_runner.status.value == "running":
        logger.info("[AutoCampaign] Campaign already running — skipping.")
        return

    telephony = "exotel"
    csv_path = "data/leads/unified_compliant_leads.csv"

    # Fallback: if unified CSV is empty/missing, use the static seed file
    from pathlib import Path as _Path
    if not _Path(csv_path).exists() or _Path(csv_path).stat().st_size < 100:
        csv_path = "data/leads/hni_leads_pune.csv"
        logger.info("[AutoCampaign] unified_compliant_leads.csv not ready — using seed file")

    logger.info(f"[AutoCampaign] Starting morning campaign via Exotel — csv={csv_path}")
    result = await campaign_runner.start(
        bot_type="investment",
        csv_path=csv_path,
        gap_seconds=90,
        max_calls=50,
        enforce_optimal_windows=True,
    )
    logger.info(f"[AutoCampaign] Campaign started: {result}")


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


class MakeCallRequest(BaseModel):
    """Request model for the make-call endpoint."""
    phone: str  # e.g. "9022873952" or "+919022873952"
    bot_type: str = "investment"  # "investment", "insurance", or "recruitment"
    customer_name: Optional[str] = None  # Pass name for personalized calls
    category: Optional[str] = None  # Pass CRM category for testing the welcome hook


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


@app.post("/api/trigger-scrape")
async def trigger_scrape():
    """Manually trigger the lead scraper (for testing)."""
    await run_nightly_scrape()
    return {"status": "Scrape completed"}


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
# Campaign Automation API
# -------------------------------------------------------

class StartCampaignRequest(BaseModel):
    """Request model for starting a calling campaign."""
    bot_type: str = "investment"  # "investment", "insurance", or "recruitment"
    csv_path: str = "data/leads/unified_compliant_leads.csv"
    gap_seconds: int = 90  # Seconds between calls
    max_calls: int = 50
    enforce_optimal_windows: bool = True  # Only call during 10-12 AM, 3-5 PM


@app.post("/api/start-campaign")
async def start_campaign(request: StartCampaignRequest):
    """
    Start an automated calling campaign.

    Reads leads from CSV, deduplicates against call logs,
    then calls each lead sequentially via Exotel.
    Enforces TRAI-compliant calling hours (9 AM – 9 PM).

    Body:
        bot_type: "investment" | "insurance" | "recruitment"
        csv_path: Path to leads CSV (default: data/leads/hni_leads_pune.csv)
        gap_seconds: Delay between calls (default: 90)
        max_calls: Max calls in this campaign (default: 50)
        enforce_optimal_windows: Only call during golden hours (default: true)
    """
    result = await campaign_runner.start(
        bot_type=request.bot_type,
        csv_path=request.csv_path,
        gap_seconds=request.gap_seconds,
        max_calls=request.max_calls,
        enforce_optimal_windows=request.enforce_optimal_windows,
    )
    status_code = 200 if "error" not in result else 400
    return JSONResponse(result, status_code=status_code)


@app.post("/api/stop-campaign")
async def stop_campaign():
    """Stop the currently running campaign after the current call completes."""
    result = campaign_runner.stop()
    status_code = 200 if "error" not in result else 400
    return JSONResponse(result, status_code=status_code)


@app.get("/api/campaign-status")
async def get_campaign_status():
    """Get current campaign status: progress, results, and calling window info."""
    return campaign_runner.get_status()


@app.get("/api/calling-status")
async def calling_status():
    """
    Check if now is a good time to call.
    Returns TRAI compliance, optimal window status, and seconds until next window.
    """
    return get_calling_status()


# -------------------------------------------------------
# Exotel Voice AI Pipeline
# -------------------------------------------------------

@app.websocket("/exotel/stream")
async def exotel_stream(websocket: WebSocket):
    """
    WebSocket endpoint that receives raw PCM audio from Exotel Voicebot Applet,
    and bridges it to Deepgram, Groq, and AWS Polly.
    """
    await websocket.accept()
    logger.info("[Exotel] WebSocket connected.")

    manager = ExotelVoiceConnectionManager(websocket)

    try:
        while True:
            data = await websocket.receive()
            if data["type"] == "websocket.disconnect":
                logger.info("[Exotel] WebSocket connection closed normally.")
                break
                
            message = None
            if "text" in data:
                message = data["text"]
            elif "bytes" in data:
                try:
                    message = data["bytes"].decode("utf-8")
                except UnicodeDecodeError:
                    logger.warning("[Exotel] Received binary audio data directly without JSON wrapper")
                    continue
                    
            if message:
                await manager.handle_exotel_message(message)
    except WebSocketDisconnect:
        logger.info("[Exotel] WebSocket connection closed normally.")
    except Exception as e:
        logger.error(f"[Exotel] WebSocket error: {e}")


@app.post("/api/make-call-exotel")
async def make_call_exotel(payload: MakeCallRequest):
    """
    Trigger an outbound call using Exotel REST API.
    The call connects to the Voicebot Applet flow which streams audio
    to our /exotel/stream WebSocket endpoint.

    Body:
        phone: str — phone number (e.g. "+919022873952" or "9022873952")
        bot_type: str — "riya" (default)
    """
    if not config.EXOTEL_API_KEY or not config.EXOTEL_API_TOKEN:
        return JSONResponse(
            {"status": "error", "message": "Exotel credentials missing."},
            status_code=400
        )

    # Normalize phone number to a valid E.164 dial string (+91XXXXXXXXXX).
    # Handles malformed inputs like '+9108087594750' (stray leading 0 after +91).
    from server.campaign.trai_compliance import to_dial_format
    phone = to_dial_format(payload.phone)
    if not phone:
        return JSONResponse(
            {"status": "error", "message": f"Invalid phone number: {payload.phone!r}"},
            status_code=400,
        )

    # Exotel REST API with Basic Auth in URL
    url = (
        f"https://{config.EXOTEL_API_KEY}:{config.EXOTEL_API_TOKEN}"
        f"@{config.EXOTEL_SUBDOMAIN}/v1/Accounts/{config.EXOTEL_ACCOUNT_SID}"
        f"/Calls/connect.json"
    )

    form_data = {
        "From": phone,
        "CallerId": config.EXOTEL_CALLER_ID,
        "Url": f"http://my.exotel.com/{config.EXOTEL_ACCOUNT_SID}/exoml/start_voice/{config.EXOTEL_APP_ID}",
        "CallType": "trans",
    }
    
    # Pass bot_type, customer_name, and category back to us via CustomField
    custom_params = {"bot_type": payload.bot_type}
    if payload.customer_name:
        custom_params["customer_name"] = payload.customer_name
    if payload.category:
        custom_params["category"] = payload.category
    
    import json
    form_data["CustomField"] = json.dumps(custom_params)

    try:
        response = requests.post(url, data=form_data, timeout=30)
        logger.info(f"[Exotel] Call API response ({response.status_code}): {response.text[:300]}")

        if response.status_code == 200:
            return {"status": "initiated", "exotel_response": response.json()}
        else:
            return JSONResponse(
                {"status": "error", "message": response.text[:500]},
                status_code=response.status_code
            )
    except Exception as e:
        logger.error(f"[Exotel] Call API error: {e}")
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )


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
