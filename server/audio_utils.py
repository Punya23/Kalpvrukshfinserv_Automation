import base64
import audioop

def pcm_to_mulaw(pcm_bytes: bytes) -> str:
    """
    Converts 16-bit 8000Hz PCM audio bytes to 8-bit mu-law base64 encoded string.
    This is the exact format Twilio Media Streams requires.
    """
    # Convert 16-bit PCM to 8-bit mu-law
    ulaw_bytes = audioop.lin2ulaw(pcm_bytes, 2)
    # Twilio expects a base64 encoded string of the payload
    return base64.b64encode(ulaw_bytes).decode("ascii")
