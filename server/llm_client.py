"""
Shared LLM clients + a resilient, LOW-LATENCY completion helper.

Provider chain (fastest → most available):

  Primary : Cerebras   — ultra-fast free inference (~2-5x faster than Groq)
                         Models: llama-3.3-70b, llama3.1-8b, llama3.1-70b
                         Base URL: https://api.cerebras.ai/v1 (OpenAI-compatible)
  Fallback: OpenRouter — no daily quota, routes to fastest available provider
                         (Cerebras/Groq/SambaNova) via throughput sort
  Last    : Groq 70B   — separate free tier, kept as final safety net
            Groq 8B    — highest free daily cap, absolute last resort

Circuit-breaker on Cerebras: if it hits its rate-limit, skip it for 3 min
so we don't waste ~0.2s probing a known-throttled endpoint every turn.

Both the voice pipeline and the post-call scorer import `complete()` from here.
"""

import asyncio
import logging
import time

import httpx
from openai import AsyncOpenAI

from server.config import config

logger = logging.getLogger(__name__)

_http = httpx.AsyncClient(
    limits=httpx.Limits(max_connections=10, max_keepalive_connections=5, keepalive_expiry=120),
    timeout=httpx.Timeout(30.0, connect=10.0),
)

# ── Cerebras (primary) ────────────────────────────────────────────────────────
cerebras_client = (
    AsyncOpenAI(
        api_key=config.CEREBRAS_API_KEY,
        base_url=config.CEREBRAS_BASE_URL,
        http_client=_http,
    )
    if config.CEREBRAS_API_KEY
    else None
)

# ── OpenRouter (fallback) ─────────────────────────────────────────────────────
openrouter_client = (
    AsyncOpenAI(
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL,
        http_client=_http,
    )
    if config.OPENROUTER_API_KEY
    else None
)

# ── Groq (last resort) ────────────────────────────────────────────────────────
groq_client = None
if config.GROQ_API_KEY:
    try:
        from groq import AsyncGroq
        groq_client = AsyncGroq(api_key=config.GROQ_API_KEY, http_client=_http)
    except ImportError:
        logger.warning("[LLM] groq package not installed — Groq fallback unavailable")

GROQ_PRIMARY_MODEL  = config.LLM_MODEL or "llama-3.3-70b-versatile"
GROQ_FALLBACK_MODEL = "llama-3.1-8b-instant"   # separate, higher free daily cap

# Circuit-breaker state (monotonic timestamps)
_COOLDOWN_SECS = 180
_cerebras_cooldown_until  = 0.0
_openrouter_cooldown_until = 0.0


def _attempt_chain() -> list[tuple]:
    """
    Returns ordered (label, model, client, extra_kwargs) tuples.
    Both free Cerebras models are tried before falling back to paid providers.

    Order:
      1. Cerebras gpt-oss-120b  (primary — 120B, best quality)
      2. Cerebras gemma-4-31b   (secondary — 31B, fast fallback if 120B rate-limited)
      3. OpenRouter              (no daily quota, routes to fastest available)
      4. Groq 70B                (free tier)
      5. Groq 8B                 (highest free daily cap, last resort)
    """
    now = time.monotonic()
    chain = []

    # 1. Cerebras primary (gpt-oss-120b)
    if cerebras_client and now >= _cerebras_cooldown_until:
        chain.append(("cerebras", config.CEREBRAS_MODEL, cerebras_client, {}))
        # 2. Cerebras fallback (gemma-4-31b) — same client, different model
        if config.CEREBRAS_MODEL_FALLBACK and config.CEREBRAS_MODEL_FALLBACK != config.CEREBRAS_MODEL:
            chain.append(("cerebras-fallback", config.CEREBRAS_MODEL_FALLBACK, cerebras_client, {}))

    # 3. OpenRouter fallback — ask for fastest provider via throughput sort
    if openrouter_client and now >= _openrouter_cooldown_until:
        chain.append((
            "openrouter",
            config.OPENROUTER_MODEL,
            openrouter_client,
            {"extra_body": {"provider": {"sort": "throughput"}}},
        ))

    # 4. Groq 70B
    if groq_client:
        chain.append(("groq", GROQ_PRIMARY_MODEL, groq_client, {}))
        # 5. Groq 8B (highest daily cap — absolute last resort)
        if GROQ_FALLBACK_MODEL != GROQ_PRIMARY_MODEL:
            chain.append(("groq-8b", GROQ_FALLBACK_MODEL, groq_client, {}))

    return chain


async def complete(messages, temperature: float = 0.5, max_tokens: int = 150):
    """Low-latency chat completion with provider fallback. Raises only if all providers fail."""
    global _cerebras_cooldown_until, _openrouter_cooldown_until

    chain = _attempt_chain()
    if not chain:
        raise RuntimeError(
            "No LLM provider configured — set CEREBRAS_API_KEY, OPENROUTER_API_KEY, or GROQ_API_KEY"
        )

    last_err = None
    last_empty_resp = None  # kept as a last-resort return if every provider comes back empty
    for i, (label, model, client, extra) in enumerate(chain):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **extra,
            )

            # Reasoning models (e.g. Cerebras gpt-oss-120b) can burn the ENTIRE token
            # budget on their hidden reasoning trace before writing any `content` —
            # especially with a long system prompt. That's a successful API call with
            # a useless empty reply, which the except-block below would never catch.
            # Treat it as a soft failure and fall through to the next (non-reasoning)
            # provider, unless this is the last one — then return it as-is and let the
            # caller's message_content()/fallback-text path handle the empty content.
            content = (resp.choices[0].message.content or "").strip()
            is_last = i == len(chain) - 1
            if not content and not is_last:
                last_empty_resp = resp
                logger.warning(
                    f"[LLM] {label}:{model} returned empty content "
                    f"(finish_reason={resp.choices[0].finish_reason}, likely reasoning-token exhaustion) "
                    "— trying next provider"
                )
                await asyncio.sleep(0.1)
                continue

            if i > 0:
                logger.warning(f"[LLM] served by fallback {label}:{model}")
            else:
                logger.debug(f"[LLM] {label}:{model} OK")
            return resp

        except Exception as e:
            last_err = e
            err_str = str(e)
            is_rate = "429" in err_str or "rate_limit" in err_str.lower() or "quota" in err_str.lower()

            if label == "cerebras" and is_rate:
                _cerebras_cooldown_until = time.monotonic() + _COOLDOWN_SECS
                logger.warning(
                    f"[LLM] Cerebras rate-limited — cooling down {_COOLDOWN_SECS}s, "
                    f"falling back to OpenRouter"
                )
            elif label == "openrouter" and is_rate:
                _openrouter_cooldown_until = time.monotonic() + _COOLDOWN_SECS
                logger.warning(f"[LLM] OpenRouter rate-limited — cooling down {_COOLDOWN_SECS}s")
            else:
                logger.warning(
                    f"[LLM] {label}:{model} failed "
                    f"({'rate-limit' if is_rate else 'error'}): {err_str[:120]}"
                )

            await asyncio.sleep(0.2)

    # Every provider either errored or came back empty. Prefer a real exception if we
    # have one; otherwise every provider "succeeded" with empty content (reasoning-token
    # exhaustion) — return the last such response so the caller's message_content()/
    # fallback-text path can degrade gracefully instead of us raising None.
    if last_err is not None:
        raise last_err
    return last_empty_resp


def message_content(response, default: str = "") -> str:
    """Safely extract assistant text from a chat completion (content may be None)."""
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        return default
    return (content or default).strip()
