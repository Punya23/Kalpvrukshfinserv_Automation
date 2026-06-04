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


def chunk_pcm(pcm_bytes: bytes, chunk_size: int = 3200) -> list[str]:
    """
    Split PCM audio into base64-encoded chunks for Exotel streaming.
    Exotel requires chunks in multiples of 320 bytes.
    Default 3200 bytes = 100ms of audio at 8kHz 16-bit mono.
    """
    chunks = []
    for i in range(0, len(pcm_bytes), chunk_size):
        chunk = pcm_bytes[i:i + chunk_size]
        chunks.append(base64.b64encode(chunk).decode("ascii"))
    return chunks
