"""
Kalpvruksh Finserv — Renewal Reminder Scheduler
Automated cron-based scheduler that checks for upcoming renewals
and triggers outbound reminder calls/messages.
"""

import asyncio
import logging
from datetime import datetime, timedelta

from server.sheets_manager import sheets_manager, whatsapp_notifier

logger = logging.getLogger(__name__)


class RenewalScheduler:
    """
    Checks for upcoming policy renewals and triggers automated reminders.
    Schedule: Runs daily at 9:00 AM IST.

    Reminder Timeline:
    - D-60: WhatsApp message (first gentle reminder)
    - D-30: WhatsApp + Phone call alert to manager
    - D-15: WhatsApp + Urgent alert to manager
    - D-7:  WhatsApp + CRITICAL alert to manager
    - D-1:  Final WhatsApp + Emergency alert
    """

    REMINDER_MILESTONES = [60, 30, 15, 7, 1]

    async def check_and_send_reminders(self):
        """Main scheduler function — checks renewals and sends appropriate reminders."""
        logger.info("🔄 Running renewal check...")

        try:
            renewals = sheets_manager.get_upcoming_renewals(days_ahead=61)
        except Exception as e:
            logger.error(f"Failed to fetch renewals: {e}")
            return

        if not renewals:
            logger.info("No upcoming renewals found.")
            return

        reminders_sent = 0
        for renewal in renewals:
            days_left = renewal.get("Days Until Expiry")
            if days_left is None:
                # Calculate from expiry date
                try:
                    expiry = datetime.strptime(renewal["Expiry Date"], "%Y-%m-%d")
                    days_left = (expiry - datetime.now()).days
                except (ValueError, KeyError):
                    continue

            # Check if this renewal matches any milestone
            for milestone in self.REMINDER_MILESTONES:
                if days_left == milestone:
                    await self._send_reminder(renewal, days_left)
                    reminders_sent += 1
                    break

            # Also handle overdue (grace period)
            if -30 <= days_left < 0:
                await self._send_overdue_alert(renewal, abs(days_left))
                reminders_sent += 1

        logger.info(f"✅ Renewal check complete. {reminders_sent} reminders sent.")

    async def _send_reminder(self, renewal: dict, days_left: int):
        """Send a renewal reminder based on the days remaining."""
        customer_name = renewal.get("Name", "Customer")
        phone = renewal.get("Phone", "")

        # Always send WhatsApp to customer
        await whatsapp_notifier.send_renewal_reminder(
            customer_phone=phone,
            customer_name=customer_name,
            policy_details=renewal,
        )

        # Alert manager based on urgency
        if days_left <= 30:
            await whatsapp_notifier.notify_manager_renewal_alert(renewal)

        logger.info(f"📋 Renewal reminder sent: {customer_name} (D-{days_left})")

    async def _send_overdue_alert(self, renewal: dict, days_overdue: int):
        """Send critical alert for overdue/lapsed policies."""
        customer_name = renewal.get("Name", "Customer")
        phone = renewal.get("Phone", "")

        # Urgent WhatsApp to customer
        overdue_message = (
            f"🔴 *URGENT* — {customer_name}!\n\n"
            f"Aapki {renewal.get('Plan', 'health insurance')} policy "
            f"{days_overdue} din pehle expire ho chuki hai.\n\n"
            f"⚠️ Grace period mein hain — abhi renew karna zaroori hai warna:\n"
            f"• Continuity benefit chala jayega\n"
            f"• No Claim Bonus reset ho jayega\n"
            f"• Naye waiting periods lagenge\n\n"
            f"Abhi call karein: {renewal.get('Phone', '')}\n"
            f"— Kalpvruksh Finserv 🌳"
        )
        await whatsapp_notifier._send_whatsapp(phone, overdue_message)

        # Critical alert to manager
        renewal["Days Until Expiry"] = -days_overdue
        await whatsapp_notifier.notify_manager_renewal_alert(renewal)

        logger.warning(f"🔴 OVERDUE ALERT: {customer_name} ({days_overdue} days past expiry)")

    async def get_renewal_summary(self) -> dict:
        """Get a summary of all upcoming renewals for the dashboard."""
        renewals = sheets_manager.get_upcoming_renewals(days_ahead=90)

        summary = {
            "total_upcoming": len(renewals),
            "due_7_days": [],
            "due_30_days": [],
            "due_60_days": [],
            "overdue": [],
            "total_premium_at_risk": 0,
        }

        for r in renewals:
            days = r.get("Days Until Expiry", 999)
            premium = r.get("Premium", 0)
            summary["total_premium_at_risk"] += premium

            entry = {
                "name": r.get("Name"),
                "customer_id": r.get("Customer ID"),
                "plan": r.get("Plan"),
                "expiry": r.get("Expiry Date"),
                "premium": premium,
                "days_left": days,
            }

            if days < 0:
                summary["overdue"].append(entry)
            elif days <= 7:
                summary["due_7_days"].append(entry)
            elif days <= 30:
                summary["due_30_days"].append(entry)
            elif days <= 60:
                summary["due_60_days"].append(entry)

        return summary


# Singleton
renewal_scheduler = RenewalScheduler()
