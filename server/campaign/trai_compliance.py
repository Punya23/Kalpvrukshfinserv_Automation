"""
Kalpvruksh Finserv — TRAI Compliance & Call Scheduling Rules
Enforces legal calling hours, optimal time windows, and lead deduplication.
"""

import logging
from datetime import datetime, time
from pathlib import Path
from typing import Optional
import json

logger = logging.getLogger(__name__)

# -------------------------------------------------------
# TRAI Legal Boundaries (TCCCPR 2018)
# -------------------------------------------------------
TRAI_START = time(9, 0)   # 9:00 AM IST — earliest allowed
TRAI_END = time(21, 0)    # 9:00 PM IST — latest allowed

# -------------------------------------------------------
# Optimal Calling Windows (based on B2B research)
# -------------------------------------------------------
OPTIMAL_WINDOWS = [
    (time(10, 0), time(12, 0)),   # Morning Gold: 10 AM – 12 PM
    (time(15, 0), time(17, 0)),   # Afternoon:    3 PM – 5 PM
]

# Best days: Tuesday (1), Wednesday (2), Thursday (3)
# Acceptable: Monday (0), Friday (4)
# Avoid: Saturday (5), Sunday (6)
BEST_DAYS = {1, 2, 3}         # Tue, Wed, Thu
ACCEPTABLE_DAYS = {0, 4}      # Mon, Fri
BLOCKED_DAYS = {5, 6}         # Sat, Sun


def is_within_trai_hours(now: Optional[datetime] = None) -> bool:
    """Check if current time is within TRAI-permitted calling hours (9 AM – 9 PM IST)."""
    now = now or datetime.now()
    return TRAI_START <= now.time() <= TRAI_END


def is_within_optimal_window(now: Optional[datetime] = None) -> bool:
    """Check if current time falls within an optimal calling window."""
    now = now or datetime.now()
    current_time = now.time()
    return any(start <= current_time <= end for start, end in OPTIMAL_WINDOWS)


def is_good_calling_day(now: Optional[datetime] = None) -> bool:
    """Check if today is a good day for outbound campaigns."""
    now = now or datetime.now()
    return now.weekday() not in BLOCKED_DAYS


def seconds_until_next_window(now: Optional[datetime] = None) -> int:
    """Calculate seconds until the next optimal calling window opens."""
    now = now or datetime.now()
    current_time = now.time()

    for start, end in OPTIMAL_WINDOWS:
        if current_time < start:
            # This window hasn't started yet today
            target = now.replace(hour=start.hour, minute=start.minute, second=0, microsecond=0)
            return int((target - now).total_seconds())

    # All windows passed today — return seconds until tomorrow's first window
    tomorrow_start = OPTIMAL_WINDOWS[0][0]
    target = now.replace(hour=tomorrow_start.hour, minute=tomorrow_start.minute, second=0, microsecond=0)
    from datetime import timedelta
    target += timedelta(days=1)
    return int((target - now).total_seconds())


def get_calling_status(now: Optional[datetime] = None) -> dict:
    """Return a human-readable status of current calling eligibility."""
    now = now or datetime.now()
    return {
        "current_time": now.strftime("%I:%M %p"),
        "current_day": now.strftime("%A"),
        "trai_compliant": is_within_trai_hours(now),
        "optimal_window": is_within_optimal_window(now),
        "good_day": is_good_calling_day(now),
        "can_call": is_within_trai_hours(now) and is_good_calling_day(now),
        "should_call": is_within_optimal_window(now) and is_good_calling_day(now),
        "seconds_until_next_window": seconds_until_next_window(now) if not is_within_optimal_window(now) else 0,
    }


# -------------------------------------------------------
# Lead Deduplication
# -------------------------------------------------------

def get_already_called_phones(call_logs_dir: str = "data/call_logs") -> set:
    """
    Scan all call log JSON files and return a set of phone numbers
    that have already been called (to prevent duplicate calls).
    """
    called = set()
    logs_path = Path(call_logs_dir)
    if not logs_path.exists():
        return called

    for log_file in logs_path.glob("*.json"):
        try:
            data = json.loads(log_file.read_text(encoding="utf-8"))
            phone = data.get("phone", "")
            if phone:
                # Normalize: strip +91, spaces, dashes
                clean = phone.replace("+", "").replace(" ", "").replace("-", "").strip()
                if clean.startswith("91") and len(clean) > 10:
                    clean = clean[2:]  # Remove country code
                called.add(clean)
        except Exception:
            continue

    logger.info(f"Found {len(called)} previously called numbers in call logs.")
    return called


def normalize_phone(phone: str) -> str:
    """Normalize an Indian phone number to 10-digit format for dedup comparison."""
    clean = phone.replace("+", "").replace(" ", "").replace("-", "").strip()
    if clean.startswith("91") and len(clean) > 10:
        clean = clean[2:]
    return clean


def deduplicate_leads(leads: list[dict], call_logs_dir: str = "data/call_logs") -> list[dict]:
    """
    Remove leads whose phone numbers have already been called.
    Returns only fresh, uncalled leads.
    """
    already_called = get_already_called_phones(call_logs_dir)
    fresh = []
    dupes = 0

    seen_in_batch = set()
    for lead in leads:
        phone = normalize_phone(lead.get("phone", ""))
        if not phone or len(phone) < 10:
            continue
        if phone in already_called:
            dupes += 1
            continue
        if phone in seen_in_batch:
            dupes += 1
            continue
        seen_in_batch.add(phone)
        fresh.append(lead)

    logger.info(f"Deduplication: {len(leads)} total → {len(fresh)} fresh, {dupes} skipped.")
    return fresh
