"""
Kalpvruksh Finserv AI Automation — Main Server
FastAPI application with webhook routes for WhatsApp, Voice AI, and testing.
"""

import json
import logging
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
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
from server import call_manager
from server.voice_pipeline import VoiceConnectionManager, ExotelVoiceConnectionManager
from twilio.rest import Client
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


class MakeCallRequest(BaseModel):
    """Request model for the make-call endpoint."""
    phone: str  # e.g. "9022873952" or "+919022873952"
    bot_type: str = "riya"  # "riya" (investment) or "aarav" (insurance)
    customer_name: Optional[str] = None  # Pass name for personalized calls


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
        function_name = body.get("message", {}).get("functionCall", {}).get("name", "")
        parameters = body.get("message", {}).get("functionCall", {}).get("parameters", {})
        logger.info(f"Vapi function call: {function_name}({parameters})")
        return JSONResponse({"result": "Function processed"})

    elif event_type == "transcript":
        transcript = body.get("message", {}).get("transcript", "")
        logger.info(f"Vapi transcript: {transcript[:100]}...")
        return JSONResponse({"status": "received"})

    elif event_type == "end-of-call-report":
        logger.info("Vapi call ended")
        return JSONResponse({"status": "noted"})

    return JSONResponse({"status": "ok"})


# -------------------------------------------------------
# Bolna AI Voice Call Endpoints
# -------------------------------------------------------

@app.post("/webhook/bolna")
async def bolna_webhook(request: Request):
    """
    Bolna AI call completion webhook.
    Receives call data when a voice call ends.
    Processes lead data, scores it, and logs to Google Sheets / local storage.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    logger.info(f"Bolna webhook received: {json.dumps(body)[:200]}...")
    record = call_manager.process_bolna_webhook(body)

    return JSONResponse({
        "status": "processed",
        "lead_status": record.get("lead_status"),
        "lead_score": record.get("lead_score"),
        "estimated_cost": record.get("estimated_cost_inr"),
    })


@app.post("/api/make-call")
async def make_call(request: MakeCallRequest):
    """
    Trigger a new voice call via Bolna AI.

    Body:
        phone: str — phone number (e.g. "9022873952")
        bot_type: str — "riya" (investment) or "aarav" (insurance)
    """
    result = await call_manager.create_agent_and_call(
        phone=request.phone,
        bot_type=request.bot_type,
    )
    status_code = 200 if result.get("status") != "error" else 400
    return JSONResponse(result, status_code=status_code)


@app.get("/api/system-status")
async def system_status():
    """
    Full system status: all APIs, models, providers, costs, and call stats.
    This is the main dashboard endpoint.
    """
    return call_manager.get_system_status()


@app.get("/api/call-history")
async def api_call_history():
    """List all voice call records with costs and lead data."""
    records = call_manager.get_call_history()
    return {
        "total": len(records),
        "calls": records,
    }


# -------------------------------------------------------
# Twilio Voice AI Pipeline (Phase 3)
# -------------------------------------------------------

@app.post("/twilio/incoming-call")
async def twilio_incoming_call(request: Request):
    """
    Twilio webhook for incoming calls. 
    Returns TwiML that connects the call to our WebSocket.
    """
    # Use TUNNEL_HOST env var or fall back to the request's Host header
    host = os.getenv("TUNNEL_HOST", request.headers.get("host", "localhost:8000"))
    # Ensure wss:// in production, ws:// for local testing
    ws_url = f"wss://{host}/twilio/stream" if "localhost" not in host and "127.0.0.1" not in host else f"ws://{host}/twilio/stream"
    
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{ws_url}" />
    </Connect>
</Response>"""
    return Response(content=twiml, media_type="text/xml")


@app.websocket("/twilio/stream")
async def twilio_stream(websocket: WebSocket):
    """
    WebSocket endpoint that receives raw audio from Twilio Media Streams,
    and bridges it to Deepgram, Groq, and AWS Polly.
    """
    await websocket.accept()
    logger.info("Twilio WebSocket connected.")
    
    manager = VoiceConnectionManager(websocket)
    
    try:
        while True:
            data = await websocket.receive()
            if data["type"] == "websocket.disconnect":
                logger.info("Twilio WebSocket connection closed normally.")
                break
            
            message = None
            if "text" in data:
                message = data["text"]
            elif "bytes" in data:
                message = data["bytes"].decode("utf-8")
                
            if message:
                await manager.handle_twilio_message(message)
    except WebSocketDisconnect:
        logger.info("Twilio WebSocket connection closed normally.")
    except Exception as e:
        logger.error(f"Twilio WebSocket error: {e}")


@app.post("/api/make-call-twilio")
async def make_call_twilio(payload: MakeCallRequest, request: Request):
    """
    Trigger an outbound call using Twilio REST API.
    The call connects to the /twilio/incoming-call webhook.
    """
    if not config.TWILIO_ACCOUNT_SID or not config.TWILIO_AUTH_TOKEN:
        return JSONResponse({"status": "error", "message": "Twilio credentials missing."}, status_code=400)
        
    client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
    
    # Use TUNNEL_HOST env var or fall back to the request's Host header
    host = os.getenv("TUNNEL_HOST", request.headers.get("host", "localhost:8000"))
    
    try:
        # Build the webhook URL dynamically based on current tunnel
        webhook_url = f"https://{host}/twilio/incoming-call" if "localhost" not in host and "127.0.0.1" not in host else f"http://{host}/twilio/incoming-call"
        
        call = client.calls.create(
            url=webhook_url,
            to=payload.phone,
            from_=getattr(config, "TWILIO_PHONE_NUMBER", "+1234567890") # Use config or placeholder
        )
        return {"status": "initiated", "call_sid": call.sid}
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get("/api/cost-summary")
async def api_cost_summary():
    """Cost breakdown: total spend, per-call averages, by bot."""
    return call_manager.get_cost_summary()


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

    # Normalize phone number
    phone = payload.phone.strip()
    if not phone.startswith("+"):
        if not phone.startswith("91"):
            phone = f"91{phone}"
        phone = f"+{phone}"

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
