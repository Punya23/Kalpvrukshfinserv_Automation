import requests
import sys

def make_call(phone: str, bot_type: str = "riya"):
    url = "http://localhost:8000/api/make-call-twilio"
    
    print(f"Triggering {bot_type} Twilio bot call to {phone}...")
    try:
        response = requests.post(url, json={"phone": phone, "bot_type": bot_type})
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        phone = sys.argv[1]
    else:
        phone = "+919370079820" # Default test number
        
    make_call(phone)
