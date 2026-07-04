import base64


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
