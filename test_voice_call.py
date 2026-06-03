"""
Kalpvruksh Finserv — Voice Call Test Script
Makes an actual outbound AI voice call to a phone number using Vapi.ai

SETUP REQUIRED BEFORE RUNNING:
1. Sign up at https://dashboard.vapi.ai
2. Get your API key from Dashboard → API Keys
3. IMPORTANT FOR INDIA (+91): You need to import a phone number via:
   - Twilio (with India SIP trunk) OR
   - Vapi's "Import via SIP" feature
4. Get a Sarvam AI API key from https://console.sarvam.ai (free ₹1000 credits)
5. Fill in the values below and run: python test_voice_call.py

ALTERNATIVE (EASIEST FOR INDIA):
Use Bolna AI — see test_voice_call_bolna.py
"""

import requests
import sys
import os

# ============================================
# CONFIGURATION — Fill these in before running
# ============================================

VAPI_API_KEY = os.getenv("VAPI_API_KEY", "YOUR_VAPI_API_KEY_HERE")

# The phone number ID from your Vapi dashboard (must be an imported Indian number)
VAPI_PHONE_NUMBER_ID = os.getenv("VAPI_PHONE_NUMBER_ID", "YOUR_PHONE_NUMBER_ID_HERE")

# The number to call (E.164 format)
CUSTOMER_PHONE = "+919284078848"

# ============================================
# Insurance Bot (Aarav) — Transient Assistant
# ============================================

# Load the insurance prompt
PROMPT_FILE = os.path.join(os.path.dirname(__file__), "prompts", "insurance_bot_prompt.txt")
if os.path.exists(PROMPT_FILE):
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()
else:
    SYSTEM_PROMPT = """You are Aarav, an insurance advisor at Kalpvruksh Finserv, Pune.
You speak naturally in Hinglish. Your goal is to understand the customer's health insurance needs
and if they are interested, forward their details to your senior manager Sanjeev sir.
Be warm, professional, and helpful. Never sound robotic."""


def make_call_with_vapi():
    """Make an outbound call using Vapi.ai with a transient assistant."""

    url = "https://api.vapi.ai/call"

    headers = {
        "Authorization": f"Bearer {VAPI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "phoneNumberId": VAPI_PHONE_NUMBER_ID,
        "customer": {
            "number": CUSTOMER_PHONE
        },
        "assistant": {
            # First message the bot speaks when call connects
            "firstMessage": (
                "Namaste! Main Aarav bol raha hoon, Kalpvruksh Finserv se, Pune. "
                "Kya aapke paas 2 minute hain? Main aapko ek important cheez batana chahta tha "
                "health insurance ke baare mein."
            ),

            # LLM Configuration
            "model": {
                "provider": "groq",
                "model": "llama-3.3-70b-versatile",
                "systemPrompt": SYSTEM_PROMPT,
                "temperature": 0.7,
            },

            # Voice Configuration — Using Sarvam AI for natural Hindi
            # Option A: Sarvam AI (best for Hindi/Hinglish)
            # Requires custom TTS server — see below
            # Option B: ElevenLabs (good multilingual support)
            "voice": {
                "provider": "11labs",
                "voiceId": "pNInz6obpgDQGcFmaJgB",  # "Adam" voice — natural male
                # For Hindi-specific, use a cloned or multilingual voice
            },

            # Transcriber (Speech-to-Text)
            "transcriber": {
                "provider": "deepgram",
                "model": "nova-2",
                "language": "hi",  # Hindi primary
            },

            # Call settings
            "endCallMessage": "Dhanyawaad aapka! Aapka din shubh ho. Kalpvruksh Finserv se Aarav tha.",
            "silenceTimeoutSeconds": 15,
            "maxDurationSeconds": 300,  # 5 min max

            # Function calling tools
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "forward_to_manager",
                        "description": "Forward hot lead details to Sanjeev sir for callback",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "customer_name": {"type": "string"},
                                "interest": {"type": "string"},
                                "summary": {"type": "string"},
                            },
                            "required": ["customer_name", "summary"],
                        },
                    },
                    # Webhook for tool execution — points to our server
                    "server": {
                        "url": "https://your-server.com/webhook/vapi"  # Replace with ngrok URL for testing
                    }
                }
            ],
        }
    }

    print(f"📞 Initiating call to {CUSTOMER_PHONE}...")
    print(f"🤖 Bot: Aarav (Insurance Advisor)")
    print(f"🧠 LLM: Groq (Llama 3.3 70B)")
    print(f"=" * 50)

    try:
        response = requests.post(url, json=payload, headers=headers)

        if response.status_code == 201:
            call_data = response.json()
            print(f"✅ Call initiated successfully!")
            print(f"   Call ID: {call_data.get('id', 'N/A')}")
            print(f"   Status: {call_data.get('status', 'N/A')}")
            print(f"\n📱 Your phone should ring in a few seconds...")
            return call_data
        else:
            print(f"❌ Failed to initiate call")
            print(f"   Status: {response.status_code}")
            print(f"   Error: {response.text}")
            return None

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


if __name__ == "__main__":
    if VAPI_API_KEY == "YOUR_VAPI_API_KEY_HERE":
        print("=" * 60)
        print("⚠️  SETUP REQUIRED")
        print("=" * 60)
        print()
        print("To make an actual call, you need to:")
        print()
        print("1. Sign up at: https://dashboard.vapi.ai")
        print("2. Get your API key from Dashboard → API Keys")
        print("3. For India (+91), import a phone number via Twilio SIP")
        print("4. Set environment variables:")
        print("   set VAPI_API_KEY=your_key_here")
        print("   set VAPI_PHONE_NUMBER_ID=your_phone_id_here")
        print()
        print("EASIER ALTERNATIVE FOR INDIA:")
        print("   Run: python test_voice_call_bolna.py")
        print("   (Bolna AI natively supports +91 Indian numbers)")
        print()
        sys.exit(1)

    make_call_with_vapi()
