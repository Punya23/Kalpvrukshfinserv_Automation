"""
Kalpvruksh Finserv — TRAI Compliance & Call Scheduling Rules
Enforces legal calling hours, optimal time windows, and lead deduplication.
"""

import logging
import re
from datetime import datetime, time
from pathlib import Path
from typing import Optional
import json
import pytz

_IST = pytz.timezone("Asia/Kolkata")


def _now_ist() -> datetime:
    """Return the current time in IST. Railway runs on UTC so datetime.now()
    must never be used bare — always call this helper instead."""
    return datetime.now(_IST)

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
    now = now or _now_ist()
    return TRAI_START <= now.time() <= TRAI_END


def is_within_optimal_window(now: Optional[datetime] = None) -> bool:
    """Check if current time falls within an optimal calling window."""
    now = now or _now_ist()
    current_time = now.time()
    return any(start <= current_time <= end for start, end in OPTIMAL_WINDOWS)


def is_good_calling_day(now: Optional[datetime] = None) -> bool:
    """Check if today is a good day for outbound campaigns."""
    now = now or _now_ist()
    return now.weekday() not in BLOCKED_DAYS


def seconds_until_next_window(now: Optional[datetime] = None) -> int:
    """Calculate seconds until the next optimal calling window opens."""
    now = now or _now_ist()
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
    now = now or _now_ist()
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
                # Use the shared normalizer so dedup keys match the dial path
                clean = normalize_phone(phone)
                if clean:
                    called.add(clean)
        except Exception:
            continue

    logger.info(f"Found {len(called)} previously called numbers in call logs.")
    return called


def normalize_phone(phone: str) -> str:
    """Normalize an Indian phone number to bare 10-digit format for dedup comparison.

    Handles messy inputs like '+9108087594750' (a stray leading 0 after the
    country code), '08087594750', '918087594750', spaces/dashes, etc.
    Always returns the last 10 significant digits, e.g. '8087594750'.
    """
    digits = re.sub(r"\D", "", phone or "")
    # Strip the 91 country code when present (handles 11+ digit strings)
    if digits.startswith("91") and len(digits) > 10:
        digits = digits[2:]
    # Strip any leading zeros (STD-format numbers like 08087594750)
    digits = digits.lstrip("0")
    # Return the last 10 digits (guards against any remaining prefix junk)
    return digits[-10:] if len(digits) >= 10 else digits


def to_dial_format(phone: str) -> str:
    """Return an E.164 dial string ('+91XXXXXXXXXX') for a valid Indian mobile.

    Returns an empty string if the number can't be normalized to 10 digits,
    so callers can skip invalid leads instead of dialing garbage.
    """
    ten = normalize_phone(phone)
    return f"+91{ten}" if len(ten) == 10 else ""


# -------------------------------------------------------
# Per-Lead Attempt Ledger
# -------------------------------------------------------
ATTEMPTS_FILE = Path("data/call_attempts.json")


def get_attempt_counts(attempts_file: Path = ATTEMPTS_FILE) -> dict:
    """Return {normalized_phone: attempt_count} from the ledger."""
    if not attempts_file.exists():
        return {}
    try:
        data = json.loads(attempts_file.read_text(encoding="utf-8"))
        out = {}
        for k, v in data.items():
            out[k] = int(v.get("attempts", 0)) if isinstance(v, dict) else int(v)
        return out
    except Exception as e:
        logger.error(f"Error reading attempts ledger: {e}")
        return {}


def record_attempt(phone: str, outcome: str = "", attempts_file: Path = ATTEMPTS_FILE) -> int:
    """Increment the attempt count for a number and return the new total.

    Called once per placed call (any outcome). No-answer/busy/failed calls don't
    write a transcript, so this ledger is what eventually caps their retries.
    """
    num = normalize_phone(phone)
    if len(num) != 10:
        return 0
    try:
        attempts_file.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        if attempts_file.exists():
            try:
                data = json.loads(attempts_file.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        entry = data.get(num) or {}
        if not isinstance(entry, dict):
            entry = {"attempts": int(entry)}
        entry["attempts"] = int(entry.get("attempts", 0)) + 1
        entry["last_outcome"] = outcome
        entry["last_attempt"] = datetime.now().isoformat()
        data[num] = entry
        attempts_file.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return entry["attempts"]
    except Exception as e:
        logger.error(f"Error recording attempt: {e}")
        return 0


def filter_by_attempt_cap(
    leads: list[dict], max_attempts: int = 3, attempts_file: Path = ATTEMPTS_FILE
) -> list[dict]:
    """Drop leads whose number has already been dialed >= max_attempts times."""
    if max_attempts <= 0:
        return leads
    counts = get_attempt_counts(attempts_file)
    kept, capped = [], 0
    for lead in leads:
        num = normalize_phone(lead.get("phone", ""))
        if counts.get(num, 0) >= max_attempts:
            capped += 1
            continue
        kept.append(lead)
    if capped:
        logger.info(
            f"Attempt cap ({max_attempts}): removed {capped} maxed-out lead(s), "
            f"{len(kept)} remain."
        )
    return kept


# -------------------------------------------------------
# DND / Do-Not-Call List
# -------------------------------------------------------
DND_FILE = Path("data/dnd_list.csv")


def get_dnd_set(dnd_file: Path = DND_FILE) -> set:
    """Return the set of DND phone numbers (normalized to 10 digits).

    The file is a CSV with a 'phone' column; comment lines (starting with '#')
    and blank/invalid rows are ignored. Never dial anything in this set.
    """
    dnd: set = set()
    if not dnd_file.exists():
        return dnd

    try:
        with open(dnd_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # First comma-separated field is the phone number
                raw = line.split(",")[0].strip()
                if raw.lower() == "phone":  # header row
                    continue
                num = normalize_phone(raw)
                if len(num) == 10:
                    dnd.add(num)
    except Exception as e:
        logger.error(f"Error reading DND list: {e}")

    if dnd:
        logger.info(f"Loaded {len(dnd)} DND numbers.")
    return dnd


def add_to_dnd(phone: str, reason: str = "", dnd_file: Path = DND_FILE) -> bool:
    """Append a number to the DND list if not already present.

    Returns True if a new entry was written. Safe to call repeatedly.
    Used to honour a caller's request to not be contacted (DNC outcome).
    """
    num = normalize_phone(phone)
    if len(num) != 10:
        return False
    if num in get_dnd_set(dnd_file):
        return False  # already listed
    try:
        from datetime import date
        dnd_file.parent.mkdir(parents=True, exist_ok=True)
        new_file = not dnd_file.exists()
        with open(dnd_file, "a", encoding="utf-8") as f:
            if new_file:
                f.write("phone,reason,date_added\n")
            # Store in +91 form for human readability; loader re-normalizes anyway
            f.write(f"+91{num},{reason},{date.today().isoformat()}\n")
        logger.info(f"Added {num} to DND list (reason: {reason or 'n/a'}).")
        return True
    except Exception as e:
        logger.error(f"Error adding to DND list: {e}")
        return False


def scrub_dnd(leads: list[dict], dnd_file: Path = DND_FILE) -> list[dict]:
    """Remove any lead whose phone number is on the DND list."""
    dnd = get_dnd_set(dnd_file)
    if not dnd:
        return leads
    kept, removed = [], 0
    for lead in leads:
        if normalize_phone(lead.get("phone", "")) in dnd:
            removed += 1
            continue
        kept.append(lead)
    if removed:
        logger.info(f"DND scrub: removed {removed} lead(s), {len(kept)} remain.")
    return kept


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
