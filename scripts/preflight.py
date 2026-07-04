"""
Kalpvruksh Finserv — Pre-flight Check
=====================================
Run this BEFORE starting a calling campaign. It verifies, in order:

  1. All required environment variables are present.
  2. Live auth against every provider (Groq, Deepgram, AWS Polly, Exotel)
     — this catches an expired/invalid key before you waste a call.
  3. Exotel account balance (so you know you have credits).
  4. Current TRAI / optimal calling window.
  5. Lead CSV dialability (how many numbers are actually callable).

Usage:
    ./venv/bin/python scripts/preflight.py
    ./venv/bin/python scripts/preflight.py --quick     # skip live network checks
    ./venv/bin/python scripts/preflight.py --csv data/leads/hni_leads_pune.csv

Exit code is 0 only if there are no hard failures, so you can gate a
campaign on it:  ./venv/bin/python scripts/preflight.py && start-campaign
"""

import argparse
import csv
import sys
from pathlib import Path

# Ensure the project root is importable when run as `python scripts/preflight.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.config import config  # noqa: E402
from server.campaign.trai_compliance import (  # noqa: E402
    to_dial_format,
    get_calling_status,
    get_dnd_set,
)

# --- tiny status helpers -------------------------------------------------
OK, FAIL, WARN, INFO = "✅", "❌", "⚠️ ", "ℹ️ "
_hard_failures: list[str] = []
_warnings: list[str] = []


def ok(msg: str):
    print(f"  {OK} {msg}")


def fail(msg: str):
    print(f"  {FAIL} {msg}")
    _hard_failures.append(msg)


def warn(msg: str):
    print(f"  {WARN}{msg}")
    _warnings.append(msg)


def info(msg: str):
    print(f"  {INFO}{msg}")


def section(title: str):
    print(f"\n\033[1m{title}\033[0m")


# --- 1. env var presence -------------------------------------------------
def check_env():
    section("1. Environment variables")
    required = {
        "GROQ_API_KEY": config.GROQ_API_KEY,
        "DEEPGRAM_API_KEY": config.DEEPGRAM_API_KEY,
        "AWS_ACCESS_KEY_ID": config.AWS_ACCESS_KEY_ID,
        "AWS_SECRET_ACCESS_KEY": config.AWS_SECRET_ACCESS_KEY,
        "AWS_REGION": config.AWS_REGION,
        "EXOTEL_API_KEY": config.EXOTEL_API_KEY,
        "EXOTEL_API_TOKEN": config.EXOTEL_API_TOKEN,
        "EXOTEL_ACCOUNT_SID": config.EXOTEL_ACCOUNT_SID,
        "EXOTEL_CALLER_ID": config.EXOTEL_CALLER_ID,
        "EXOTEL_APP_ID": config.EXOTEL_APP_ID,
        "EXOTEL_SUBDOMAIN": config.EXOTEL_SUBDOMAIN,
    }
    for name, value in required.items():
        if value:
            # Mask everything but the last 4 chars
            shown = f"…{str(value)[-4:]}" if len(str(value)) > 4 else "set"
            ok(f"{name} ({shown})")
        else:
            fail(f"{name} is MISSING")

    # Optional-but-recommended
    for name, value in {
        "LEADS_SHEET_ID": config.LEADS_SHEET_ID,
        "WHATSAPP_API_KEY": config.WHATSAPP_API_KEY,
    }.items():
        if not value:
            warn(f"{name} not set (that feature will be disabled)")


# --- 2. live provider auth ----------------------------------------------
def check_groq():
    try:
        from groq import Groq
        client = Groq(api_key=config.GROQ_API_KEY)
        client.chat.completions.create(
            model=config.LLM_MODEL or "llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
        ok(f"Groq LLM reachable (model: {config.LLM_MODEL})")
    except Exception as e:
        fail(f"Groq auth/model failed: {str(e)[:160]}")


def check_polly():
    try:
        import boto3
        polly = boto3.client(
            "polly",
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
            region_name=config.AWS_REGION,
        )
        resp = polly.synthesize_speech(
            Text="<speak>test</speak>",
            TextType="ssml",
            OutputFormat="pcm",
            SampleRate="8000",
            VoiceId="Kajal",
            LanguageCode="hi-IN",
            Engine="neural",
        )
        if resp.get("AudioStream"):
            ok("AWS Polly reachable (Kajal / hi-IN / neural)")
        else:
            fail("AWS Polly returned no audio stream")
    except Exception as e:
        fail(f"AWS Polly failed: {str(e)[:160]}")


def check_deepgram():
    try:
        import requests
        r = requests.get(
            "https://api.deepgram.com/v1/projects",
            headers={"Authorization": f"Token {config.DEEPGRAM_API_KEY}"},
            timeout=15,
        )
        if r.status_code == 200:
            ok("Deepgram STT key valid")
        elif r.status_code in (401, 403):
            fail("Deepgram key rejected (401/403) — check DEEPGRAM_API_KEY")
        else:
            warn(f"Deepgram returned HTTP {r.status_code} (proceeding)")
    except Exception as e:
        fail(f"Deepgram check failed: {str(e)[:160]}")


def check_exotel():
    try:
        import requests
        url = (
            f"https://{config.EXOTEL_API_KEY}:{config.EXOTEL_API_TOKEN}"
            f"@{config.EXOTEL_SUBDOMAIN}/v1/Accounts/{config.EXOTEL_ACCOUNT_SID}.json"
        )
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            ok("Exotel API auth valid")
            try:
                acct = r.json().get("Account", {})
                bal = acct.get("Balance") or acct.get("balance")
                if bal is not None:
                    info(f"Exotel balance: {bal}")
            except Exception:
                pass
        elif r.status_code in (401, 403):
            fail("Exotel auth rejected (401/403) — check EXOTEL_API_KEY/TOKEN/SID/SUBDOMAIN")
        else:
            warn(f"Exotel returned HTTP {r.status_code}: {r.text[:120]}")
    except Exception as e:
        fail(f"Exotel check failed: {str(e)[:160]}")


def check_providers():
    section("2. Live provider auth")
    check_groq()
    check_deepgram()
    check_polly()
    check_exotel()


# --- 3. calling window ---------------------------------------------------
def check_window():
    section("3. Calling window (TRAI + optimal)")
    s = get_calling_status()
    info(f"Now: {s['current_time']} {s['current_day']}")
    (ok if s["trai_compliant"] else fail)(
        f"TRAI legal hours (9 AM–9 PM): {s['trai_compliant']}"
    )
    (ok if s["good_day"] else warn)(f"Good calling day (Mon–Fri): {s['good_day']}")
    if s["optimal_window"]:
        ok("Inside optimal window (10–12 / 3–5)")
    else:
        mins = s["seconds_until_next_window"] // 60
        warn(f"Outside optimal window — next opens in ~{mins} min")
    if not s["can_call"]:
        warn("can_call=False — a campaign will pause/stop until the window opens")


# --- 4. lead CSV dialability --------------------------------------------
def check_leads(csv_path: str):
    section(f"4. Lead list: {csv_path}")
    path = Path(csv_path)
    if not path.exists():
        fail(f"CSV not found: {csv_path}")
        return

    dialable, invalid, dnd_hits = 0, 0, 0
    invalid_samples: list[str] = []
    dnd = get_dnd_set()

    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = (row.get("phone") or "").strip()
            dial = to_dial_format(raw)
            if not dial:
                invalid += 1
                if len(invalid_samples) < 5:
                    invalid_samples.append(raw or "(empty)")
                continue
            if dial[-10:] in dnd:
                dnd_hits += 1
                continue
            dialable += 1

    (ok if dialable else fail)(f"Dialable leads: {dialable}")
    if invalid:
        warn(f"Invalid/undialable rows skipped: {invalid} → e.g. {invalid_samples}")
    if dnd_hits:
        info(f"On DND, will be scrubbed: {dnd_hits}")
    if dialable == 0:
        fail("No dialable leads — nothing to call")


# --- main ----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Kalpvruksh campaign pre-flight check")
    ap.add_argument("--csv", default="data/leads/hni_leads_pune.csv", help="Leads CSV path")
    ap.add_argument("--quick", action="store_true", help="Skip live network auth checks")
    args = ap.parse_args()

    print("\033[1m🌳 Kalpvruksh Finserv — Pre-flight Check\033[0m")

    check_env()
    if not args.quick:
        check_providers()
    else:
        section("2. Live provider auth")
        info("skipped (--quick)")
    check_window()
    check_leads(args.csv)

    # Summary
    section("Summary")
    if _hard_failures:
        print(f"  {FAIL} {len(_hard_failures)} blocker(s) — DO NOT start the campaign:")
        for m in _hard_failures:
            print(f"     - {m}")
    if _warnings:
        print(f"  {WARN}{len(_warnings)} warning(s) (non-blocking):")
        for m in _warnings:
            print(f"     - {m}")
    if not _hard_failures and not _warnings:
        print(f"  {OK} All clear — cleared for takeoff. 🚀")
    elif not _hard_failures:
        print(f"  {OK} No blockers. Review warnings above, then you're good to call.")

    sys.exit(1 if _hard_failures else 0)


if __name__ == "__main__":
    main()
