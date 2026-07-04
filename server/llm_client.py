"""
Shared LLM clients + a resilient, LOW-LATENCY completion helper.

Ordering matters for a voice bot: Groq's LPU replies in ~0.2-0.3s, whereas OpenRouter's
70B routes through slower providers (often 1-3s). So:

  Primary : Groq 70B  (fastest — used whenever its daily quota is available)
  Fallback: OpenRouter (no daily quota; forced to its FASTEST provider via throughput sort)
  Last    : Groq 8B    (separate, higher free quota — final safety net)

A circuit-breaker skips Groq 70B for a cooldown after it hits its daily cap, so we don't
waste ~0.3s probing a known-exhausted model on every turn during a campaign.

Both the voice pipeline and the post-call scorer import `complete()` from here.
This module only depends on `config`, so it can't create an import cycle.
"""

import asyncio
import logging
import time

import httpx
from groq import AsyncGroq
from openai import AsyncOpenAI

from server.config import config

logger = logging.getLogger(__name__)

_http = httpx.AsyncClient(
    limits=httpx.Limits(max_connections=10, max_keepalive_connections=5, keepalive_expiry=120),
    timeout=httpx.Timeout(30.0, connect=10.0),
)

groq_client = (
    AsyncGroq(api_key=config.GROQ_API_KEY, http_client=_http) if config.GROQ_API_KEY else None
)
openrouter_client = (
    AsyncOpenAI(api_key=config.OPENROUTER_API_KEY, base_url=config.OPENROUTER_BASE_URL, http_client=_http)
    if config.OPENROUTER_API_KEY
    else None
)

GROQ_FALLBACK_MODEL = "llama-3.1-8b-instant"  # separate, higher free daily cap
_GROQ_COOLDOWN_SECS = 180
_groq_primary_cooldown_until = 0.0  # monotonic time until which Groq 70B is skipped


def _attempt_chain():
    """Ordered (label, model, client) attempts — fast Groq first, OpenRouter as no-quota backup."""
    now = time.monotonic()
    chain = []
    groq_primary = (config.LLM_MODEL or "llama-3.3-70b-versatile") if groq_client else None
    if groq_primary and now >= _groq_primary_cooldown_until:
        chain.append(("groq", groq_primary, groq_client))
    if openrouter_client:
        chain.append(("openrouter", config.OPENROUTER_MODEL, openrouter_client))
    if groq_client and groq_primary != GROQ_FALLBACK_MODEL:
        chain.append(("groq", GROQ_FALLBACK_MODEL, groq_client))
    return chain


async def complete(messages, temperature: float = 0.5, max_tokens: int = 150):
    """Low-latency chat completion with provider fallback. Raises only if all providers fail."""
    global _groq_primary_cooldown_until
    chain = _attempt_chain()
    if not chain:
        raise RuntimeError("No LLM provider configured — set GROQ_API_KEY or OPENROUTER_API_KEY")

    last_err = None
    for i, (label, model, client) in enumerate(chain):
        try:
            # When falling back to OpenRouter, ask it for the FASTEST provider for this model
            # (Cerebras/Groq/SambaNova serve llama-3.3-70b at very high throughput).
            extra = {"extra_body": {"provider": {"sort": "throughput"}}} if label == "openrouter" else {}
            resp = await client.chat.completions.create(
                model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, **extra
            )
            if i > 0:
                logger.warning(f"[LLM] served by fallback {label}:{model}")
            return resp
        except Exception as e:
            last_err = e
            is_rate = "429" in str(e) or "rate_limit" in str(e).lower()
            if label == "groq" and model != GROQ_FALLBACK_MODEL and is_rate:
                # Fast primary is out of daily quota — skip it for a while instead of probing every turn.
                _groq_primary_cooldown_until = time.monotonic() + _GROQ_COOLDOWN_SECS
                logger.warning(f"[LLM] Groq primary rate-limited — cooling down {_GROQ_COOLDOWN_SECS}s, using OpenRouter")
            else:
                logger.warning(f"[LLM] {label}:{model} failed ({'rate-limit' if is_rate else 'error'}): {str(e)[:100]}")
            await asyncio.sleep(0.2)
    raise last_err
