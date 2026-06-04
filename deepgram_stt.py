import sys
import os
import requests
from server.config import config

def transcribe_audio(file_path):
    url = "https://api.deepgram.com/v1/listen?model=nova-2&language=hi&smart_format=true"
    headers = {
        "Authorization": f"Token {config.DEEPGRAM_API_KEY}",
        "Content-Type": "audio/mp3"
    }
    
    try:
        with open(file_path, "rb") as file:
            response = requests.post(url, headers=headers, data=file)
            
        if response.status_code == 200:
            data = response.json()
            transcript = data.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("transcript", "")
            print("--- TRANSCRIPT START ---")
            print(transcript)
            print("--- TRANSCRIPT END ---")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    file_path = "/Users/punyasurana/Documents/Kalpvrukshfinserv_Automation/8fe51548065e1e44be830302816c1a64.mp3"
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    transcribe_audio(file_path)
