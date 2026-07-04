# Exotel Voicebot Setup — Kalpvruksh Finserv

This is the exact path from "I have an Exotel number" to "the AI bot is talking to a
lead." The code is already written (`/exotel/stream` WebSocket + `ExotelVoiceConnectionManager`
in `server/voice_pipeline.py`). Everything below is **configuration**, not code.

Run the preflight after each step to see what's still red:

```bash
./venv/bin/python scripts/preflight.py
```

---

## Step 0 — Exotel API credentials ✅ RESOLVED

The 403/401 was a typo in `.env`: `EXOTEL_ACCOUNT_SID=ykalpvrukshfinserv1` (stray leading
`y`). Corrected to `kalpvrukshfinserv1`. Auth now returns HTTP 200, and the account reports
`KycStatus: completed`. Verified config:
- Account SID: `kalpvrukshfinserv1`  ·  Region: Singapore  ·  Subdomain: `api.exotel.com`

## Step 0.5 — TRIAL ACCOUNT LIMIT (CURRENT COMMERCIAL BLOCKER 🚫)

The account is `Type: Trial`. **Exotel trial accounts can only call phone numbers that are
added as *Verified Numbers* in the dashboard.** So:
- ✅ You CAN test the full pipeline **right now** by verifying your own mobile and calling it.
- 🚫 You CANNOT dial the 44 scraped leads until you **upgrade to a paid account** (add credits
  / request activation — KYC is already done, so this is mainly funding).

To verify a test number: Exotel Dashboard → **Manage → Verified Caller IDs / Numbers** → add
and OTP-verify your own mobile. Then use it in the Step 4 smoke test.

---

## Step 1 — Confirm Voicebot / bidirectional streaming is enabled

The `/exotel/stream` socket only receives audio if your account has the **Voicebot
(bidirectional media streaming)** feature. It is **not on by default** — it's a
premium capability Exotel enables per account.

- In App Bazaar (Step 3), can you add an applet called **"Voicebot"** / **"Voice Streaming"**?
  - **Yes** → you're enabled, continue.
  - **No** → raise a ticket with your Exotel account manager: *"Please enable bidirectional
    voice streaming (Voicebot applet) on account `<SID>`."* This is the item that can take
    a day, so start it first.

---

## Step 2 — Expose the server on a public `wss://` URL

Exotel's cloud cannot reach `localhost`. Pick one:

**A. Fastest, for testing — ngrok**
```bash
# terminal 1: run the server
./venv/bin/python -m server.main
# terminal 2: tunnel it
ngrok http 8000
```
ngrok prints `https://<random>.ngrok-free.app`. Your stream URL is:
```
wss://<random>.ngrok-free.app/exotel/stream
```

**B. Production — Railway** (config already in repo: `railway.json` + `Procfile`)
- Deploy the repo, set all `.env` vars in Railway's dashboard.
- Your stream URL is `wss://<your-app>.up.railway.app/exotel/stream`.

> The stream URL lives inside the Exotel applet (Step 3), **not** in `.env`. When the
> ngrok URL changes (free tier rotates it), update the applet.

---

## Step 3 — Build the App flow with the Voicebot applet

1. Exotel Dashboard → **App Bazaar → Create App** (or edit the existing one whose ID is
   in `EXOTEL_APP_ID`).
2. Drag in the **Voicebot** applet.
3. Set its **WebSocket URL** to your `wss://…/exotel/stream` from Step 2.
4. Save. Copy the **App SID/ID** from the URL or app list → put it in `.env` as
   `EXOTEL_APP_ID` (yours currently ends `…1314`).
5. Make sure your ExoPhone/CallerId (`EXOTEL_CALLER_ID`, ends `…6363`) is attached to
   this App for outbound.

> Audio format the code already assumes (must match the applet): **8 kHz, 16-bit,
> mono, little-endian PCM (raw/SLIN)**, base64 in JSON frames. Don't set the applet to
> mu-law or 16 kHz.

---

## Step 4 — Smoke-test ONE call to your own phone

Do **not** run the campaign first. With the server running and preflight green:

```bash
curl -X POST http://localhost:8000/api/make-call-exotel \
  -H "Content-Type: application/json" \
  -d '{"phone":"<your-own-number>","bot_type":"investment"}'
```

Watch the server logs. Success looks like:
```
[Exotel] WebSocket connected.
[Exotel] Stream started. SID: ...
[Exotel] Connected to Deepgram STT (linear16).
[Exotel] Welcoming customer directly: ...
[Exotel] Sending outbound chunk 1 ...
[Exotel] User: <what you said>
[Exotel] Bot: <reply>
```

### If it connects but something's wrong
- **Bot silent / dead air** → ✅ FIXED. The outbound frame now uses `stream_sid` (snake_case),
  matching Exotel AgentStream.
  Confirmed against Exotel's echobot example: `{"event":"media","stream_sid":...,"media":{"payload":...}}`.
- **Bot uses the wrong persona / no name** → the `CustomField` (bot_type/name/category) isn't
  arriving in the stream `start` event. Harmless: the pipeline falls back to a phone-number
  lookup in `data/leads/hni_leads_pune.csv`, which works now that dialing is normalized.

---

## Step 5 — Run a small campaign

Only after a clean smoke test. Start with a tiny batch, during an optimal window
(Tue–Thu, 10–12 or 3–5 — **not** weekends; the runner will pause otherwise):

```bash
curl -X POST http://localhost:8000/api/start-campaign \
  -H "Content-Type: application/json" \
  -d '{"bot_type":"investment","csv_path":"data/leads/hni_leads_pune.csv","telephony_provider":"exotel","max_calls":3,"gap_seconds":90}'

# monitor
curl http://localhost:8000/api/campaign-status
# stop early if needed
curl -X POST http://localhost:8000/api/stop-campaign
```

The runner dedups against `data/call_logs/`, scrubs `data/dnd_list.csv`, enforces TRAI
hours, and writes a transcript + QA row per call. Anyone who says "don't call me" is
auto-added to the DND list and never dialed again.

---

## Compliance reminder ⚠️
You're calling numbers scraped from Google Maps. In India, commercial voice calls are
governed by **TRAI TCCCPR / DLT**. Hours are enforced in code, but keep the DND list
current (`data/dnd_list.csv`) and confirm with Sanjeev that the outreach basis is sound
before scaling past a test batch.

---

## Quick reference — endpoints
| Endpoint | Purpose |
|---|---|
| `POST /api/make-call-exotel` | Single outbound call (smoke test) |
| `WS /exotel/stream` | Bidirectional audio (Exotel applet points here) |
| `POST /api/start-campaign` | Batch calling from CSV |
| `GET /api/campaign-status` | Live campaign progress |
| `POST /api/stop-campaign` | Stop after current call |
| `GET /api/calling-status` | Is now a good time to call? |
