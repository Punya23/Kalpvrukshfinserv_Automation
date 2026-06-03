"""
Kalpvruksh Finserv — Make an actual voice call via Bolna AI
Testing Riya (Investment Bot)
"""

import requests
import json
import os
import sys

# Fix Windows terminal encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

BOLNA_API_KEY = os.getenv("BOLNA_API_KEY", "")
BOLNA_BASE_URL = "https://api.bolna.ai"

# Changed to 9022873952
CUSTOMER_PHONE = "+919284078848"

# Load investment prompt
PROMPT_FILE = os.path.join(os.path.dirname(__file__), "prompts", "investment_bot_prompt.txt")
if os.path.exists(PROMPT_FILE):
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()
else:
    SYSTEM_PROMPT = (
        "You are Riya, an investment advisor at Kalpvruksh Finserv, Pune. "
        "Speak naturally in Hinglish. Help customers with mutual funds, SIPs, and investments. "
        "Be warm, professional, and never sound robotic."
    )


def create_agent_and_call():
    """Create a Bolna agent and immediately make a call."""

    headers = {
        "Authorization": f"Bearer {BOLNA_API_KEY}",
        "Content-Type": "application/json"
    }

    agent_payload = {
        "agent_config": {
            "agent_name": "Riya - Kalpvruksh Investment Bot",
            "agent_type": "other",
            "agent_welcome_message": "Hi, namaste! Main Riya bol rahi hoon Kalpvruksh Finserv se. Kya aapse do minute baat ho sakti hai?",
            "tasks": [
                {
                    "task_type": "conversation",
                    "toolchain": {
                        "execution": "parallel",
                        "pipelines": [["transcriber", "llm", "synthesizer"]]
                    },
                    "tools_config": {
                        "llm_agent": {
                            "agent_type": "simple_llm_agent",
                            "agent_flow_type": "streaming",
                            "llm_config": {
                                "provider": "openai",
                                "family": "openai",
                                "model": "gpt-4o-mini",
                                "max_tokens": 150,
                                "temperature": 0.5,
                                "agent_flow_type": "streaming",
                            },
                        },
                        "transcriber": {
                            "provider": "deepgram",
                            "model": "nova-2",
                            "language": "hi",
                            "stream": True,
                            "endpointing": 250,
                        },
                        "synthesizer": {
                            "provider": "sarvam",
                            "provider_config": {
                                "voice": "priya",
                                "model": "bulbul:v3",
                                "language": "hi-IN"
                            },
                            "stream": True,
                            "buffer_size": 100,
                            "audio_format": "wav",
                        },
                        "input": {
                            "provider": "twilio",
                            "format": "wav",
                        },
                        "output": {
                            "provider": "twilio",
                            "format": "wav",
                        },
                    },
                    "task_config": {
                        "hangup_after_silence": 20,
                        "incremental_delay": 100,
                        "number_of_words_for_interruption": 1,
                        "call_terminate": 180,
                        "backchanneling": True,
                        "backchanneling_message_gap": 5,
                        "backchanneling_start_delay": 4,
                    },
                }
            ],
        },
        "agent_prompts": {
            "task_1": {
                "system_prompt": SYSTEM_PROMPT
            }
        },
    }

    print("=" * 60)
    print("  KALPVRUKSH FINSERV - Voice Bot Test Call (Riya)")
    print("=" * 60)
    print(f"  Target:  {CUSTOMER_PHONE}")
    print(f"  Bot:     Riya (Investment Advisor)")
    print(f"  Ears:    Deepgram (Nova-2, Hindi)")
    print(f"  Brain:   Groq (Llama 3.3 70B)")
    print(f"  Voice:   AWS Polly (Kajal - Hindi Female)")
    print("=" * 60)
    print()

    # Create agent
    print("[1/2] Creating voice agent...")
    create_url = f"{BOLNA_BASE_URL}/v2/agent"
    response = requests.post(create_url, json=agent_payload, headers=headers)

    if response.status_code in (200, 201):
        agent_data = response.json()
        agent_id = agent_data.get("agent_id", agent_data.get("id", ""))
        print(f"  OK - Agent created: {agent_id}")
    else:
        print(f"  FAILED - Status {response.status_code}")
        print(f"  Response: {response.text}")
        return None

    # Make the call
    print()
    print(f"[2/2] Calling {CUSTOMER_PHONE}...")
    call_payload = {
        "agent_id": agent_id,
        "recipient_phone_number": CUSTOMER_PHONE,
    }

    call_url = f"{BOLNA_BASE_URL}/call"
    response = requests.post(call_url, json=call_payload, headers=headers)

    if response.status_code in (200, 201):
        call_data = response.json()
        print(f"  OK - Call initiated!")
        print(f"  Call ID: {call_data.get('call_id', call_data.get('id', 'N/A'))}")
        print()
        print("  >>> Your phone should ring in a few seconds! <<<")
        return call_data
    else:
        print(f"  FAILED - Status {response.status_code}")
        print(f"  Response: {response.text}")
        return None


if __name__ == "__main__":
    if not BOLNA_API_KEY:
        print("ERROR: BOLNA_API_KEY not found in .env")
        sys.exit(1)
    create_agent_and_call()
