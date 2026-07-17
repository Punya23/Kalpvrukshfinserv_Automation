"""
Kalpvruksh Finserv AI Automation — Google Sheets Manager
Handles read/write operations to Google Sheets for lead tracking,
renewal management, and manager notifications.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx

from server.config import config
from server.lead_scoring import LeadData, LeadCategory

logger = logging.getLogger(__name__)

# -------------------------------------------------------
# Try to import gspread; if not available, use a mock
# (allows the server to start without Google credentials)
# -------------------------------------------------------
try:
    import gspread
    from google.oauth2.service_account import Credentials

    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False
    logger.warning("gspread not installed — Google Sheets integration disabled. Using local JSON fallback.")


class SheetsManager:
    """Manages Google Sheets operations for lead and renewal tracking."""

    def __init__(self):
        self._client = None
        self._spreadsheet = None
        self._warned_no_creds = False  # warn once, not on every call
        self._local_fallback_dir = config.DATA_DIR / "local_sheets"
        self._local_fallback_dir.mkdir(parents=True, exist_ok=True)

    def _get_client(self):
        """Lazy-load Google Sheets client."""
        if self._client is not None:
            return self._client

        if not GSPREAD_AVAILABLE:
            return None

        try:
            scopes = [
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive",
            ]
            creds_path = Path(config.GOOGLE_SHEETS_CREDENTIALS_FILE)
            client_secret_path = Path("client_secret.json")
            auth_user_path = Path("credentials/authorized_user.json")
            
            # Prefer OAuth Client ID if the user generated a token
            if auth_user_path.exists() and client_secret_path.exists():
                self._client = gspread.oauth(
                    credentials_filename=str(client_secret_path),
                    authorized_user_filename=str(auth_user_path)
                )
                return self._client
                
            # Fallback to Service Account if the user managed to bypass GCP policy
            if creds_path.exists():
                credentials = Credentials.from_service_account_file(str(creds_path), scopes=scopes)
                self._client = gspread.authorize(credentials)
                return self._client
                
            if not self._warned_no_creds:
                logger.warning(
                    "No Google Sheets credentials found (need authorized_user.json or "
                    "service_account.json) — using local JSON fallback for all leads."
                )
                self._warned_no_creds = True
            return None
            
        except Exception as e:
            logger.error(f"Failed to authenticate with Google Sheets: {e}")
            return None

    def _get_spreadsheet(self):
        """Get the spreadsheet instance."""
        if self._spreadsheet is not None:
            return self._spreadsheet

        client = self._get_client()
        if client is None or not config.LEADS_SHEET_ID:
            return None

        try:
            self._spreadsheet = client.open_by_key(config.LEADS_SHEET_ID)
            return self._spreadsheet
        except Exception as e:
            logger.error(f"Failed to open spreadsheet: {e}")
            return None

    def _write_to_local_fallback(self, sheet_name: str, row_data: dict):
        """Fallback: write to local JSON file if Google Sheets is unavailable."""
        filepath = self._local_fallback_dir / f"{sheet_name.replace(' ', '_').lower()}.json"

        existing = []
        if filepath.exists():
            try:
                existing = json.loads(filepath.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                # Don't silently overwrite — that would discard every previously
                # logged lead. Preserve the corrupt file for manual recovery, then
                # start fresh so this write still succeeds.
                backup = filepath.with_suffix(
                    f".corrupt-{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
                try:
                    filepath.rename(backup)
                    logger.error(f"Corrupt fallback JSON — preserved as {backup}, starting fresh.")
                except Exception as e:
                    logger.error(f"Corrupt fallback JSON and backup failed: {e}")
                existing = []
        if not isinstance(existing, list):
            existing = [existing] if existing else []

        existing.append(row_data)
        filepath.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Written to local fallback: {filepath}")

    def log_hot_lead(self, lead: LeadData, scheduled_day: str = None, scheduled_time: str = None) -> bool:
        """
        Log a hot lead to the 'Hot Leads' sheet.
        Schema: Timestamp | Name | Phone | Age | Occupation | Family Size |
                Currently Insured | Interest | Budget | Score | Category |
                Bot | Source | Summary | Scheduled Callback | Manager Action | Status
        """
        callback_str = f"{scheduled_day} {scheduled_time}".strip() if scheduled_day else "N/A"
        row_data = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Name": lead.name,
            "Phone": lead.phone,
            "Age": lead.age or "N/A",
            "Occupation": lead.occupation or "N/A",
            "Family Members": lead.family_members,
            "Currently Insured": "Yes" if lead.currently_insured else "No",
            "Interest": lead.insurance_interest or lead.financial_goal or "General",
            "Budget/Surplus": str(lead.investable_surplus or "N/A"),
            "Lead Score": lead.score,
            "Category": lead.category.value,
            "Bot": lead.bot_type.value.title(),
            "Source": lead.source.value,
            "Conversation Summary": lead.conversation_summary[:500],
            "Scheduled Callback": callback_str,
            "Manager Action": "PENDING CALLBACK",
            "Status": "NEW",
        }

        spreadsheet = self._get_spreadsheet()
        if spreadsheet is None:
            self._write_to_local_fallback(config.LEADS_SHEET_NAME, row_data)
            return True

        try:
            worksheet = spreadsheet.worksheet(config.LEADS_SHEET_NAME)
            worksheet.append_row(list(row_data.values()))
            logger.info(f"Hot lead logged: {lead.name} (Score: {lead.score})")
            return True
        except Exception as e:
            logger.error(f"Failed to write to Google Sheets: {e}")
            self._write_to_local_fallback(config.LEADS_SHEET_NAME, row_data)
            return True

    def log_nurture_lead(self, lead: LeadData) -> bool:
        """Log a warm lead to the 'Nurture Pipeline' sheet."""
        row_data = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Name": lead.name,
            "Phone": lead.phone,
            "Interest": lead.insurance_interest or lead.financial_goal or "General",
            "Score": lead.score,
            "Bot": lead.bot_type.value.title(),
            "Next Follow-up": "7 days",
            "Notes": lead.conversation_summary[:300],
            "Status": "NURTURING",
        }

        spreadsheet = self._get_spreadsheet()
        if spreadsheet is None:
            self._write_to_local_fallback(config.NURTURE_SHEET_NAME, row_data)
            return True

        try:
            worksheet = spreadsheet.worksheet(config.NURTURE_SHEET_NAME)
            worksheet.append_row(list(row_data.values()))
            logger.info(f"Nurture lead logged: {lead.name} (Score: {lead.score})")
            return True
        except Exception as e:
            logger.error(f"Failed to write nurture lead: {e}")
            self._write_to_local_fallback(config.NURTURE_SHEET_NAME, row_data)
            return True

    def get_upcoming_renewals(self, days_ahead: int = 60) -> list[dict]:
        """Fetch policies due for renewal in the next N days."""
        spreadsheet = self._get_spreadsheet()
        if spreadsheet is None:
            logger.warning("Google Sheets unavailable — no renewals fetched. Skipping.")
            return []

        try:
            worksheet = spreadsheet.worksheet(config.RENEWALS_SHEET_NAME)
            records = worksheet.get_all_records()
            
            today = datetime.now().date()
            filtered = []
            for r in records:
                try:
                    expiry_str = r.get("Expiry Date", "").strip()
                    if not expiry_str:
                        continue
                    expiry = datetime.strptime(expiry_str, "%Y-%m-%d").date()
                    days_left = (expiry - today).days
                    if -30 <= days_left <= days_ahead:
                        r["Days Until Expiry"] = days_left
                        filtered.append(r)
                except (ValueError, TypeError):
                    continue
            return filtered
        except Exception as e:
            logger.error(f"Failed to read renewals: {e}")
            return []


class WhatsAppNotifier:
    """Send WhatsApp notifications to the manager and customers."""

    def __init__(self):
        self._client = httpx.AsyncClient(timeout=30.0)

    async def notify_manager_hot_lead(self, lead: LeadData, scheduled_day: str = None, scheduled_time: str = None) -> bool:
        """Send WhatsApp alert to Sanjeev sir about a hot lead."""
        callback_line = ""
        if scheduled_day:
            callback_line = f"\n⏰ *Requested Callback:* {scheduled_day} {scheduled_time or ''}".strip()

        message = (
            f"🔴 *NEW HOT LEAD — {lead.bot_type.value.upper()}*\n\n"
            f"👤 *Name:* {lead.name}\n"
            f"📞 *Phone:* {lead.phone}\n"
            f"🎯 *Score:* {lead.score}/10\n"
            f"📋 *Interest:* {lead.insurance_interest or lead.financial_goal}\n"
            f"💬 *Summary:* {lead.conversation_summary[:200]}"
            f"{callback_line}\n\n"
            f"⚡ *Action Required:* Call back within 1 hour"
        )

        return await self._send_whatsapp(config.MANAGER_WHATSAPP_NUMBER, message)

    @staticmethod
    def _fmt_money(value) -> str:
        """Format a premium as ₹1,23,000-style money, tolerant of str/None values.

        get_all_records() often returns numeric cells as strings (e.g. '5000'),
        and f"{'5000':,}" raises ValueError — so coerce to int first, and fall
        back to the raw value if it isn't numeric."""
        try:
            return f"{int(float(str(value).replace(',', '').strip())):,}"
        except (TypeError, ValueError):
            return str(value) if value not in (None, "") else "N/A"

    async def notify_manager_renewal_alert(self, renewal: dict) -> bool:
        """Alert manager about an urgent renewal."""
        days = renewal.get("Days Until Expiry", "?")
        try:
            urgency = "🔴 CRITICAL" if days <= 7 else "🟡 IMPORTANT" if days <= 30 else "🟢 NORMAL"
        except TypeError:
            urgency = "🟡 IMPORTANT"  # days wasn't numeric

        message = (
            f"{urgency} *RENEWAL ALERT*\n\n"
            f"👤 *Customer:* {renewal.get('Name', 'Unknown')} ({renewal.get('Customer ID', 'N/A')})\n"
            f"📋 *Policy:* {renewal.get('Plan', 'N/A')} — {renewal.get('Insurer', 'N/A')}\n"
            f"💰 *Premium:* ₹{self._fmt_money(renewal.get('Premium'))}\n"
            f"📅 *Expiry:* {renewal.get('Expiry Date', 'N/A')}\n"
            f"⏰ *Days Left:* {days}\n\n"
            f"📞 Call: {renewal.get('Phone', 'N/A')}"
        )

        return await self._send_whatsapp(config.MANAGER_WHATSAPP_NUMBER, message)

    async def send_renewal_reminder(self, customer_phone: str, customer_name: str,
                                     policy_details: dict) -> bool:
        """Send renewal reminder to customer via WhatsApp."""
        message = (
            f"Namaste {customer_name}! 🙏\n\n"
            f"Kalpvruksh Finserv se Vikram bol raha hoon.\n\n"
            f"Aapki *{policy_details.get('Plan', 'insurance')}* policy "
            f"(No: {policy_details.get('Policy Number', 'N/A')}) ka renewal "
            f"*{policy_details.get('Expiry Date', 'N/A')}* ko due hai.\n\n"
            f"💰 Renewal Premium: *₹{self._fmt_money(policy_details.get('Premium'))}*\n\n"
            f"Renew karne ke liye 'YES' reply karein ya humein call karein.\n\n"
            f"— Kalpvruksh Finserv 🌳"
        )

        return await self._send_whatsapp(customer_phone, message)

    async def _send_whatsapp(self, phone: str, message: str) -> bool:
        """Send a WhatsApp message via the configured API."""
        if not config.WHATSAPP_API_KEY or not config.WHATSAPP_API_URL:
            logger.info(f"[MOCK WhatsApp] To: {phone}\n{message}\n---")
            return True

        try:
            response = await self._client.post(
                config.WHATSAPP_API_URL,
                headers={
                    "Authorization": f"Basic {config.WHATSAPP_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "countryCode": "+91",
                    "phoneNumber": phone,
                    "type": "Text",
                    "data": {"message": message},
                },
            )
            if response.status_code == 200:
                logger.info(f"WhatsApp sent to {phone}")
                return True
            else:
                logger.error(f"WhatsApp API error: {response.status_code} — {response.text}")
                return False
        except Exception as e:
            logger.error(f"WhatsApp send failed: {e}")
            return False


# Singleton instances
sheets_manager = SheetsManager()
whatsapp_notifier = WhatsAppNotifier()
