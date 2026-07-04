"""
Kalpvruksh Finserv — Offline Call Simulator
============================================
Drives the REAL VoiceStateMachine + LLM + brevity/TTS-text guard exactly like
server/voice_pipeline.process_llm_and_speak does, but with scripted user turns
instead of Deepgram/Twilio audio. Use this to check prompt changes (brevity,
human-ness, pivot behavior, script-mixing) BEFORE placing a real phone call.

Usage:
    ./venv/bin/python scripts/simulate_call.py                       # default scenarios
    ./venv/bin/python scripts/simulate_call.py A_engaged,B_hesitant  # pick scenarios
    ./venv/bin/python scripts/simulate_call.py INS_refusal           # insurance bot

Available scenarios: A_engaged, B_hesitant, C_refusal (investment bot),
INS_refusal (insurance bot).
"""

import asyncio
import re
import sys
from pathlib import Path

# Ensure the project root is importable when run as `python scripts/simulate_call.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.voice_state_machine import VoiceStateMachine, CallState  # noqa: E402
from server.llm_client import complete as _llm_complete  # noqa: E402
from server.voice_pipeline import _clean_speak_text  # noqa: E402

# Phrases that reveal a recruitment/extra-income pivot leaking into a non-recruitment call.
PIVOT_MARKERS = re.compile(
    r"partnership|partner\b|referral|extra income|advisory income|franchise|"
    r"commission|recruit|associate|कमीशन|पार्टनरशिप|रेफरल|extra कमाई|इनकम का मौका|income का",
    re.IGNORECASE,
)


async def run_call(bot_type, user_turns, customer_name="Amit", label=""):
    sm = VoiceStateMachine(bot_type=bot_type, customer_name=customer_name, customer_category="")
    # Simulate the welcome already spoken (pipeline does this before the first user turn).
    sm.state = CallState.OPENING
    sm.chat_history.append({"role": "assistant", "content": f"नमस्ते {customer_name} जी, मैं Riya, Kalpvruksh Finserv Pune से।"})

    print(f"\n{'='*70}\n{label}  [bot={bot_type}]\n{'='*70}")
    pivot_hits = []
    transcript = []
    for i, user_text in enumerate(user_turns):
        print(f"\n👤 USER : {user_text}")
        sm.chat_history.append({"role": "user", "content": user_text})
        sm.full_transcript.append({"role": "user", "content": user_text})
        if len(sm.chat_history) > 12:
            sm.chat_history = [sm.chat_history[0]] + sm.chat_history[-10:]

        instruction = sm.get_instruction_for_current_state(user_text=user_text)
        messages = list(sm.chat_history) + [{"role": "system", "content": instruction}]
        resp = await _llm_complete(messages, temperature=0.5, max_tokens=100)
        bot_text = resp.choices[0].message.content.strip()
        speak = _clean_speak_text(bot_text)
        sm.post_process_response(bot_text)
        sm.chat_history.append({"role": "assistant", "content": speak or bot_text})
        sm.full_transcript.append({"role": "assistant", "content": bot_text})

        # Pivot detection only makes sense for non-recruitment bots — if we're
        # ALREADY testing the recruitment bot, income/commission talk is correct
        # content, not a leaked pivot.
        flag = " ⚠️PIVOT" if bot_type != "recruitment" and PIVOT_MARKERS.search(speak) else ""
        if flag:
            pivot_hits.append((i, speak))
        wc = len(speak.split())
        print(f"🤖 BOT  : {speak}   [{wc}w, state={sm.state.value}]{flag}")
        transcript.append(speak)
        if sm.state == CallState.HANGUP or "[CALL_END]" in bot_text:
            print("   (call ended)")
            break
    if bot_type == "recruitment":
        verdict = "✅ recruitment bot (pivot check N/A)"
    else:
        verdict = "❌ PIVOTED to recruitment" if pivot_hits else "✅ stayed on-product"
    print(f"\n   → {verdict}")
    return pivot_hits, transcript


# Scripted personas (realistic Hinglish replies a Pune HNI would give)
SCENARIOS = {
    "A_engaged": [
        "हाँ बोलिए", "FD में रखा है ज़्यादातर", "अच्छा, कितना return मिल सकता है SIP में?",
        "हम्म ठीक है", "हाँ मिल सकते हैं", "kal shaam 5 baje",
    ],
    "B_hesitant": [
        "हाँ", "बैंक में ही रखते हैं", "अभी सोचा नहीं है", "देखेंगे बाद में",
        "अभी टाइम नहीं है", "हम्म",
    ],
    "C_refusal": [
        "हाँ कौन", "FD है", "नहीं मुझे investment में interest नहीं है",
        "नहीं भाई पैसा नहीं है अभी", "नहीं चाहिए",
    ],
}

INS_REFUSAL = [
    "हाँ बोलिए", "corporate cover है office से", "नहीं मुझे और insurance नहीं चाहिए",
    "पहले से policy है", "नहीं चाहिए भाई",
]

REC_ENGAGED = [
    "हाँ बोलिए", "मैं CA हूँ", "अच्छा, कितना commission मिलता है?",
    "ठीक है सुनना चाहूंगा", "हाँ बता दीजिए", "kal subah theek hai",
]


async def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "B_hesitant,C_refusal"
    for name in which.split(","):
        name = name.strip()
        if name == "INS_refusal":
            await run_call("insurance", INS_REFUSAL, label="INS_refusal", customer_name="Rahul")
        elif name == "REC_engaged":
            await run_call("recruitment", REC_ENGAGED, label="REC_engaged", customer_name="Priya")
        else:
            await run_call("investment", SCENARIOS[name], label=name)


if __name__ == "__main__":
    asyncio.run(main())
