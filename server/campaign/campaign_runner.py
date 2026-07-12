"""
Kalpvruksh Finserv — Campaign Runner
Automated sequential calling engine that reads leads from CSV,
calls them one-by-one via Exotel, and tracks results.
"""

import csv
import json
import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from enum import Enum

import requests
from twilio.rest import Client as TwilioClient

from server.config import config
from server.campaign.trai_compliance import (
    is_within_trai_hours,
    is_within_optimal_window,
    is_good_calling_day,
    seconds_until_next_window,
    deduplicate_leads,
    normalize_phone,
    to_dial_format,
    scrub_dnd,
    filter_by_attempt_cap,
    record_attempt,
    get_calling_status,
)

logger = logging.getLogger(__name__)


class CampaignStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    WAITING_FOR_WINDOW = "waiting_for_window"
    COMPLETED = "completed"
    STOPPED = "stopped"
    ERROR = "error"


class CampaignRunner:
    """
    Sequential campaign engine.

    Flow:
    1. Load leads from CSV
    2. Deduplicate against call logs
    3. For each lead:
       a. Check TRAI compliance (time window)
       b. Trigger Exotel call via REST API
       c. Wait for call to complete (polling call logs)
       d. Log result
       e. Wait gap_seconds before next call
    4. Report summary
    """

    def __init__(self):
        self.status: CampaignStatus = CampaignStatus.IDLE
        self.current_campaign_id: Optional[str] = None
        self.leads: list[dict] = []
        self.results: list[dict] = []
        self.current_index: int = 0
        self.bot_type: str = "investment"
        # Read from env so Railway can flip between providers without a redeploy.
        # Set TELEPHONY_PROVIDER=exotel in Railway env vars once Exotel KYC is done.
        self.telephony_provider: str = os.getenv("TELEPHONY_PROVIDER", "twilio")
        self.gap_seconds: int = 90  # Gap between calls
        self.max_calls: int = 50
        self.enforce_optimal_windows: bool = True
        self._stop_requested: bool = False
        self._current_task: Optional[asyncio.Task] = None

    def get_status(self) -> dict:
        """Return current campaign status."""
        return {
            "campaign_id": self.current_campaign_id,
            "status": self.status.value,
            "bot_type": self.bot_type,
            "total_leads": len(self.leads),
            "calls_made": self.current_index,
            "calls_remaining": len(self.leads) - self.current_index,
            "results_summary": self._get_results_summary(),
            "calling_status": get_calling_status(),
        }

    def _get_results_summary(self) -> dict:
        """Summarize campaign results so far."""
        if not self.results:
            return {"total": 0}

        outcomes = {}
        for r in self.results:
            outcome = r.get("outcome", "unknown")
            outcomes[outcome] = outcomes.get(outcome, 0) + 1

        return {
            "total": len(self.results),
            "outcomes": outcomes,
        }

    def load_leads_from_csv(self, csv_path: str = "data/leads/hni_leads_pune.csv") -> int:
        """Load leads from CSV file and deduplicate."""
        path = Path(csv_path)
        if not path.exists():
            logger.error(f"Leads CSV not found: {csv_path}")
            return 0

        raw_leads = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    phone = row.get("phone", "").strip()
                    if phone and len(normalize_phone(phone)) >= 10:
                        raw_leads.append({
                            "name": row.get("name", "").strip(),
                            "phone": phone,
                            "category": row.get("category", "").strip(),
                            "address": row.get("address", "").strip(),
                            "rating": row.get("rating", "").strip(),
                        })
        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
            return 0

        # Gate the list: skip already-answered numbers (dedup), then DND opt-outs,
        # then numbers that have hit the retry cap (no-answers that never connected).
        self.leads = filter_by_attempt_cap(
            scrub_dnd(deduplicate_leads(raw_leads)),
            config.MAX_CALL_ATTEMPTS,
        )

        # Enforce max calls limit
        if len(self.leads) > self.max_calls:
            self.leads = self.leads[:self.max_calls]
            logger.info(f"Capped leads to {self.max_calls}")

        logger.info(f"Loaded {len(self.leads)} leads for campaign.")
        return len(self.leads)

    async def start(
        self,
        bot_type: str = "investment",
        csv_path: str = "data/leads/unified_compliant_leads.csv",
        gap_seconds: int = 90,
        max_calls: int = 50,
        enforce_optimal_windows: bool = True,
        telephony_provider: Optional[str] = None,
    ) -> dict:
        """Start the campaign. Runs as a background asyncio task."""
        if self.status == CampaignStatus.RUNNING:
            return {"error": "Campaign already running", "status": self.get_status()}

        self.bot_type = bot_type
        # Allow caller to override provider per-campaign (e.g. auto-campaign always uses env default)
        if telephony_provider:
            self.telephony_provider = telephony_provider
        self.gap_seconds = gap_seconds
        self.max_calls = max_calls
        self.enforce_optimal_windows = enforce_optimal_windows
        self.current_index = 0
        self.results = []
        self._stop_requested = False
        self.current_campaign_id = f"campaign_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Load leads
        count = self.load_leads_from_csv(csv_path)
        if count == 0:
            self.status = CampaignStatus.ERROR
            return {"error": "No leads to call", "status": self.get_status()}

        # Launch background task
        self.status = CampaignStatus.RUNNING
        self._current_task = asyncio.create_task(self._run_campaign())

        logger.info(f"🚀 Campaign {self.current_campaign_id} started: {count} leads, bot={bot_type}")
        return {"message": f"Campaign started with {count} leads", "status": self.get_status()}

    def stop(self) -> dict:
        """Request campaign to stop after current call completes."""
        if self.status != CampaignStatus.RUNNING:
            return {"error": "No campaign running"}

        self._stop_requested = True
        self.status = CampaignStatus.STOPPED
        logger.info(f"🛑 Campaign stop requested. Will stop after current call.")
        return {"message": "Campaign stop requested", "status": self.get_status()}

    async def _run_campaign(self):
        """Main campaign loop — processes leads sequentially."""
        try:
            for i, lead in enumerate(self.leads):
                if self._stop_requested:
                    logger.info("Campaign stopped by user.")
                    break

                self.current_index = i

                # --- TRAI Compliance Check ---
                if not is_within_trai_hours():
                    logger.warning("⚠️ Outside TRAI hours (9 AM – 9 PM). Stopping campaign.")
                    self.status = CampaignStatus.STOPPED
                    break

                # --- Optimal Window Check ---
                if self.enforce_optimal_windows and not is_within_optimal_window():
                    wait_secs = seconds_until_next_window()
                    if wait_secs > 7200:  # More than 2 hours — stop, don't wait
                        logger.info(f"Next window is {wait_secs // 60} min away. Pausing campaign.")
                        self.status = CampaignStatus.PAUSED
                        break
                    else:
                        logger.info(f"⏳ Waiting {wait_secs // 60} min for next optimal window...")
                        self.status = CampaignStatus.WAITING_FOR_WINDOW
                        await asyncio.sleep(wait_secs)
                        self.status = CampaignStatus.RUNNING

                # --- Day Check ---
                if not is_good_calling_day():
                    logger.info("📅 Weekend detected. Pausing campaign.")
                    self.status = CampaignStatus.PAUSED
                    break

                # --- Make the Call ---
                result = await self._make_single_call(lead)
                self.results.append(result)

                # Record the attempt (skip rows we never actually dialed, e.g. invalid phone)
                if result.get("call_status") != "skipped":
                    total = record_attempt(lead.get("phone", ""), result.get("outcome", ""))
                    result["attempt_number"] = total

                logger.info(
                    f"📞 Call {i + 1}/{len(self.leads)}: "
                    f"{lead.get('name', 'Unknown')} ({lead.get('phone')}) → {result.get('outcome', '?')}"
                )

                # --- Wait before next call ---
                if i < len(self.leads) - 1 and not self._stop_requested:
                    logger.info(f"⏱️ Waiting {self.gap_seconds}s before next call...")
                    await asyncio.sleep(self.gap_seconds)

            # Campaign complete
            if not self._stop_requested:
                self.status = CampaignStatus.COMPLETED
                self.current_index = len(self.leads)

            # Save campaign summary
            self._save_campaign_summary()
            logger.info(f"✅ Campaign {self.current_campaign_id} finished. {len(self.results)} calls made.")

        except Exception as e:
            logger.error(f"Campaign error: {e}")
            self.status = CampaignStatus.ERROR

    async def _make_single_call(self, lead: dict) -> dict:
        """Trigger a single call via configured telephony provider and wait for completion."""
        raw_phone = lead.get("phone", "").strip()
        name = lead.get("name", "").strip()
        category = lead.get("category", "").strip()

        # Normalize to a valid E.164 dial string (+91XXXXXXXXXX).
        # Handles malformed CSV values like '+9108087594750' (stray leading 0).
        phone = to_dial_format(raw_phone)
        if not phone:
            logger.warning(f"[Campaign] Skipping invalid phone {raw_phone!r} for {name}")
            return {
                "phone": raw_phone,
                "name": name,
                "category": category,
                "bot_type": self.bot_type,
                "telephony": "exotel",
                "timestamp": datetime.now().isoformat(),
                "outcome": "invalid_phone",
                "call_status": "skipped",
            }

        # Track which call logs exist BEFORE the call
        call_logs_before = set(Path("data/call_logs").glob("*.json"))

        call_result = {
            "phone": phone,
            "name": name,
            "category": category,
            "bot_type": self.bot_type,
            "telephony": self.telephony_provider,
            "timestamp": datetime.now().isoformat(),
            "outcome": "unknown",
            "call_status": "unknown",
        }

        try:
            if self.telephony_provider == "twilio":
                call_result = await self._initiate_twilio_call(phone, name, category, call_result)
            else:
                call_result = await self._initiate_exotel_call(phone, name, category, call_result)

            if call_result.get("outcome") == "api_error":
                return call_result

        except Exception as e:
            call_result["call_status"] = "failed"
            call_result["outcome"] = "api_error"
            call_result["error"] = str(e)
            logger.error(f"[Campaign] Call failed: {e}")
            return call_result

        # --- Wait for call to complete ---
        if self.telephony_provider == "twilio":
            # Twilio: just wait for a new transcript log to appear (no status polling API)
            call_result["outcome"] = await self._wait_for_transcript(
                call_logs_before, timeout_seconds=300
            )
        else:
            # Exotel: fast status polling (detects no-answer/busy in seconds)
            call_result["outcome"] = await self._wait_for_exotel_completion(
                call_result.get("call_sid"), call_logs_before
            )

        return call_result

    async def _initiate_twilio_call(self, phone: str, name: str, category: str, call_result: dict) -> dict:
        """Trigger a call via Twilio REST API (used during testing before Exotel KYC)."""
        if not getattr(config, "TWILIO_ACCOUNT_SID", None) or not getattr(config, "TWILIO_AUTH_TOKEN", None):
            call_result["call_status"] = "failed"
            call_result["outcome"] = "api_error"
            call_result["error"] = "Twilio credentials missing in env"
            logger.error("[Campaign/Twilio] TWILIO_ACCOUNT_SID or TWILIO_AUTH_TOKEN not set.")
            return call_result

        client = TwilioClient(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
        host = os.getenv("TUNNEL_HOST", "")
        if not host:
            call_result["call_status"] = "failed"
            call_result["outcome"] = "api_error"
            call_result["error"] = "TUNNEL_HOST env var not set — set it to your Railway public URL"
            logger.error("[Campaign/Twilio] TUNNEL_HOST not set.")
            return call_result

        webhook_url = f"https://{host}/twilio/incoming-call"

        try:
            call = client.calls.create(
                url=webhook_url,
                to=phone,
                from_=getattr(config, "TWILIO_PHONE_NUMBER", ""),
            )
            call_result["call_status"] = "initiated"
            call_result["call_sid"] = call.sid
            logger.info(f"[Campaign/Twilio] Call initiated → {name} ({phone}) | SID: {call.sid}")
        except Exception as e:
            call_result["call_status"] = "failed"
            call_result["outcome"] = "api_error"
            call_result["error"] = str(e)
            logger.error(f"[Campaign/Twilio] Call failed: {e}")

        return call_result

    async def _wait_for_transcript(self, logs_before: set, timeout_seconds: int = 300) -> str:
        """
        Wait for a new call log file to appear in data/call_logs/.
        Used for Twilio where we don't have a status polling API in free tier.
        """
        call_logs_dir = Path("data/call_logs")
        elapsed = 0
        poll_interval = 5

        while elapsed < timeout_seconds:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

            new_logs = set(call_logs_dir.glob("*.json")) - logs_before
            if new_logs:
                newest = max(new_logs, key=lambda p: p.stat().st_mtime)
                try:
                    data = json.loads(newest.read_text(encoding="utf-8"))
                    outcome = data.get("outcome", data.get("lead_category", "completed"))
                    logger.info(f"[Campaign/Twilio] Transcript found. Outcome: {outcome}")
                    return outcome
                except Exception:
                    return "completed"

        logger.warning("[Campaign/Twilio] No transcript within timeout — marking no-answer.")
        return "no-answer"

    async def _initiate_exotel_call(self, phone: str, name: str, category: str, call_result: dict) -> dict:
        """Trigger a call via Exotel REST API."""
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
            "CustomField": json.dumps({
                "bot_type": self.bot_type,
                "customer_name": name,
                "category": category,
            }),
        }

        response = requests.post(url, data=form_data, timeout=30)

        if response.status_code == 200:
            call_result["call_status"] = "initiated"
            # Capture the Call SID so we can poll its status (detect no-answer/busy fast).
            try:
                call_result["call_sid"] = response.json().get("Call", {}).get("Sid")
            except Exception:
                call_result["call_sid"] = None
            logger.info(
                f"[Campaign/Exotel] Call initiated to {name} ({phone}) — "
                f"SID: {call_result.get('call_sid')}"
            )
        else:
            call_result["call_status"] = "failed"
            call_result["outcome"] = "api_error"
            call_result["error"] = response.text[:200]
            logger.error(f"[Campaign/Exotel] API error: {response.status_code}")

        return call_result

    def _get_exotel_call_status(self, call_sid: str) -> Optional[str]:
        """Fetch the current Exotel status for a call (blocking; run via to_thread).

        Returns a lowercase status like 'in-progress', 'completed', 'no-answer',
        'busy', 'failed', 'ringing', 'queued' — or None if it can't be read.
        """
        if not call_sid:
            return None
        try:
            url = (
                f"https://{config.EXOTEL_API_KEY}:{config.EXOTEL_API_TOKEN}"
                f"@{config.EXOTEL_SUBDOMAIN}/v1/Accounts/{config.EXOTEL_ACCOUNT_SID}"
                f"/Calls/{call_sid}.json"
            )
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                status = r.json().get("Call", {}).get("Status")
                return status.strip().lower().replace("_", "-") if status else None
        except Exception as e:
            logger.debug(f"[Campaign/Exotel] status poll failed for {call_sid}: {e}")
        return None

    async def _wait_for_exotel_completion(
        self,
        call_sid: Optional[str],
        logs_before: set,
        connect_timeout: int = 75,
        max_call_seconds: int = 300,
        grace_after_complete: int = 20,
    ) -> str:
        """
        Wait for an Exotel call to finish, using the call's *status* as the primary
        signal. This is the fix for the no-answer stall: an unanswered / busy / failed
        call returns in seconds instead of blocking the campaign for the full timeout.

        Resolution order each poll:
          1. Transcript file appeared  -> answered & scored; return its outcome.
          2. Exotel status terminal-negative (no-answer/busy/failed/canceled) -> return it.
          3. Exotel status 'completed'  -> wait a short grace for the transcript, else 'completed'.
          4. Never reached 'in-progress' by connect_timeout -> return 'no-answer'.

        Returns: a transcript outcome, or one of
        'no-answer' | 'busy' | 'failed' | 'canceled' | 'completed' | 'timeout'.
        """
        call_logs_dir = Path("data/call_logs")
        poll_interval = 5
        elapsed = 0
        answered = False
        complete_since: Optional[int] = None

        if not call_sid:
            logger.warning(
                "[Campaign/Exotel] No Call SID returned — falling back to transcript-poll "
                "with a shorter cap (unanswered calls can't be detected without a SID)."
            )

        while elapsed < max_call_seconds:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

            # 1) Fast path — the voice pipeline wrote a transcript (call was answered)
            new_logs = set(call_logs_dir.glob("*.json")) - logs_before
            if new_logs:
                newest = max(new_logs, key=lambda p: p.stat().st_mtime)
                try:
                    data = json.loads(newest.read_text(encoding="utf-8"))
                    outcome = data.get("outcome", data.get("lead_category", "completed"))
                except Exception:
                    outcome = "completed"
                logger.info(f"[Campaign/Exotel] Transcript found. Outcome: {outcome}")
                return outcome

            # 2) Authoritative signal — Exotel call status
            status = await asyncio.to_thread(self._get_exotel_call_status, call_sid) if call_sid else None
            if status:
                if status in ("in-progress", "in-call"):
                    answered = True
                elif status in ("failed", "busy", "no-answer", "canceled", "cancelled"):
                    logger.info(f"[Campaign/Exotel] Call '{status}' (not connected) — moving on.")
                    return "canceled" if status == "cancelled" else status
                elif status == "completed":
                    # Call ended Exotel-side; give the pipeline a moment to flush its transcript
                    if complete_since is None:
                        complete_since = elapsed
                    elif elapsed - complete_since >= grace_after_complete:
                        logger.info("[Campaign/Exotel] Completed but no transcript in grace window.")
                        return "completed"
                # 'queued' / 'ringing' -> keep waiting

            # 3) Never-answered guard (covers the no-SID fallback and stuck-ringing cases)
            if not answered and elapsed >= connect_timeout:
                logger.info(f"[Campaign/Exotel] No answer within {connect_timeout}s — moving on.")
                return "no-answer"

        logger.warning("[Campaign/Exotel] Hit max wait — marking timeout.")
        return "timeout"

    def _save_campaign_summary(self):
        """Save campaign results to a JSON file."""
        summary_dir = Path("data/campaigns")
        summary_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "campaign_id": self.current_campaign_id,
            "timestamp": datetime.now().isoformat(),
            "bot_type": self.bot_type,
            "total_leads_loaded": len(self.leads),
            "total_calls_made": len(self.results),
            "results": self.results,
            "summary": self._get_results_summary(),
        }

        filename = f"{self.current_campaign_id}.json"
        filepath = summary_dir / filename
        filepath.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"📊 Campaign summary saved: {filepath}")


# Singleton
campaign_runner = CampaignRunner()
