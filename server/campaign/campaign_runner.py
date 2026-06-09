"""
Kalpvruksh Finserv — Campaign Runner
Automated sequential calling engine that reads leads from CSV,
calls them one-by-one via Exotel, and tracks results.
"""

import csv
import json
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from enum import Enum

import requests

from server.config import config
from server.campaign.trai_compliance import (
    is_within_trai_hours,
    is_within_optimal_window,
    is_good_calling_day,
    seconds_until_next_window,
    deduplicate_leads,
    normalize_phone,
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

        # Deduplicate
        self.leads = deduplicate_leads(raw_leads)

        # Enforce max calls limit
        if len(self.leads) > self.max_calls:
            self.leads = self.leads[:self.max_calls]
            logger.info(f"Capped leads to {self.max_calls}")

        logger.info(f"Loaded {len(self.leads)} leads for campaign.")
        return len(self.leads)

    async def start(
        self,
        bot_type: str = "investment",
        csv_path: str = "data/leads/hni_leads_pune.csv",
        gap_seconds: int = 90,
        max_calls: int = 50,
        enforce_optimal_windows: bool = True,
    ) -> dict:
        """Start the campaign. Runs as a background asyncio task."""
        if self.status == CampaignStatus.RUNNING:
            return {"error": "Campaign already running", "status": self.get_status()}

        self.bot_type = bot_type
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
        """Trigger a single Exotel call and wait for it to complete."""
        phone = lead.get("phone", "").strip()
        name = lead.get("name", "").strip()
        category = lead.get("category", "").strip()

        # Normalize phone
        if not phone.startswith("+"):
            if not phone.startswith("91"):
                phone = f"91{phone}"
            phone = f"+{phone}"

        # Track which call logs exist BEFORE the call
        call_logs_before = set(Path("data/call_logs").glob("*.json"))

        # Trigger call via Exotel REST API
        call_result = {
            "phone": phone,
            "name": name,
            "category": category,
            "bot_type": self.bot_type,
            "timestamp": datetime.now().isoformat(),
            "outcome": "unknown",
            "exotel_status": "unknown",
        }

        try:
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
                call_result["exotel_status"] = "initiated"
                logger.info(f"[Campaign] Call initiated to {name} ({phone})")
            else:
                call_result["exotel_status"] = "failed"
                call_result["outcome"] = "api_error"
                call_result["error"] = response.text[:200]
                logger.error(f"[Campaign] Exotel API error: {response.status_code}")
                return call_result

        except Exception as e:
            call_result["exotel_status"] = "failed"
            call_result["outcome"] = "api_error"
            call_result["error"] = str(e)
            logger.error(f"[Campaign] Call failed: {e}")
            return call_result

        # --- Wait for call to complete ---
        # Poll for new call log file (the pipeline writes one when call ends)
        call_result["outcome"] = await self._wait_for_call_completion(
            call_logs_before, timeout_seconds=300
        )

        return call_result

    async def _wait_for_call_completion(
        self, logs_before: set, timeout_seconds: int = 300
    ) -> str:
        """
        Wait until a new call log file appears in data/call_logs/.
        The voice pipeline creates one JSON file per completed call.
        """
        call_logs_dir = Path("data/call_logs")
        elapsed = 0
        poll_interval = 3  # Check every 3 seconds

        while elapsed < timeout_seconds:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

            current_logs = set(call_logs_dir.glob("*.json"))
            new_logs = current_logs - logs_before

            if new_logs:
                # A new call log appeared — read it
                newest = max(new_logs, key=lambda p: p.stat().st_mtime)
                try:
                    data = json.loads(newest.read_text(encoding="utf-8"))
                    outcome = data.get("outcome", data.get("lead_category", "unknown"))
                    logger.info(f"[Campaign] Call completed. Outcome: {outcome}")
                    return outcome
                except Exception:
                    return "completed"

        logger.warning("[Campaign] Call timed out waiting for completion.")
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
