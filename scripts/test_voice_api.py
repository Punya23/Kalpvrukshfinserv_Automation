import os
import asyncio
import boto3
from groq import AsyncGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_groq():
    print("--- Testing Groq ---")
    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    print(f"API Key: {api_key[:10]}...{api_key[-5:] if api_key else ''}")
    print(f"Model: {model}")
    
    try:
        client = AsyncGroq(api_key=api_key)
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello! Please greet me."}],
            max_tokens=20
        )
        text = response.choices[0].message.content.strip()
        print(f"Groq Response Success: '{text}'")
        return True
    except Exception as e:
        print(f"Groq Error: {e}")
        return False

def test_polly():
    print("--- Testing AWS Polly ---")
    aws_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_REGION", "ap-south-1")
    
    print(f"AWS ID: {aws_id[:6]}...{aws_id[-4:] if aws_id else ''}")
    print(f"AWS Region: {aws_region}")
    
    try:
        polly_client = boto3.client(
            "polly",
            aws_access_key_id=aws_id,
            aws_secret_access_key=aws_secret,
            region_name=aws_region
        )
        response = polly_client.synthesize_speech(
            Text="Namaste! Main Kajal bol rahi hoon Kalpvruksh Finserv se.",
            OutputFormat="pcm",
            SampleRate="8000",
            VoiceId="Kajal",
            LanguageCode="hi-IN",
            Engine="neural"
        )
        stream = response.get("AudioStream")
        data = stream.read()
        print(f"Polly Synthesis Success! Received {len(data)} bytes of PCM audio.")
        return True
    except Exception as e:
        print(f"Polly Error: {e}")
        return False

async def main():
    groq_ok = await test_groq()
    polly_ok = test_polly()
    if groq_ok and polly_ok:
        print("\nAll systems GO! Groq and AWS Polly are functioning perfectly.")
    else:
        print("\nFailures detected in systems. Please check errors above.")

if __name__ == "__main__":
    asyncio.run(main())
