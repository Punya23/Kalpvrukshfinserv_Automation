"""
Shared LLM clients + a resilient, LOW-LATENCY completion helper.

Provider chain (3 providers, tried in order):

  Primary : Groq        — fast free inference; if it exhausts its token budget
                          or errors, fall through automatically.
                          Model: llama-3.3-70b-versatile (override via GROQ_MODEL)
  Fallback: Cerebras    — ultra-fast free inference (OpenAI-compatible).
                          Base URL: https://api.cerebras.ai/v1
  Last    : OpenRouter  — no daily quota, routes to fastest available provider
                          (Cerebras/Groq/SambaNova) via throughput sort

Fall-through is automatic on either (a) an API error/rate-limit or (b) a
"successful" call that returns empty content because the model burned its whole
token budget on hidden reasoning (finish_reason=length) — see complete().

Circuit-breakers: if Groq or Cerebras hits its rate-limit, skip it for 3 min so
we don't waste ~0.2s probing a known-throttled endpoint every turn.

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

# Groq is now the primary. It needs a real Groq model — NOT config.LLM_MODEL,
# which is a Cerebras model id (gpt-oss-120b) and does not exist on Groq.
GROQ_PRIMARY_MODEL = config.GROQ_MODEL or "llama-3.3-70b-versatile"

# Circuit-breaker state (monotonic timestamps)
_COOLDOWN_SECS = 180
_groq_cooldown_until      = 0.0
_cerebras_cooldown_until  = 0.0


def _attempt_chain() -> list[tuple]:
    """
    Returns ordered (label, model, client, extra_kwargs) tuples.

    Order (3 providers):
      1. Groq       (primary)   — llama-3.3-70b-versatile
      2. Cerebras   (fallback)  — used when Groq errors or exhausts its tokens
      3. OpenRouter (last)      — no daily quota, routes to fastest available

    Providers on rate-limit cooldown are skipped until their window expires.
    """
    now = time.monotonic()
    chain = []

    # 1. Groq primary
    if groq_client and now >= _groq_cooldown_until:
        chain.append(("groq", GROQ_PRIMARY_MODEL, groq_client, {}))

    # 2. Cerebras fallback
    if cerebras_client and now >= _cerebras_cooldown_until:
        chain.append(("cerebras", config.CEREBRAS_MODEL, cerebras_client, {}))

    # 3. OpenRouter — ask for fastest provider via throughput sort
    if openrouter_client:
        chain.append((
            "openrouter",
            config.OPENROUTER_MODEL,
            openrouter_client,
            {"extra_body": {"provider": {"sort": "throughput"}}},
        ))

    return chain


async def complete(messages, temperature: float = 0.5, max_tokens: int = 150):
    """Low-latency chat completion with provider fallback. Raises only if all providers fail."""
    global _groq_cooldown_until, _cerebras_cooldown_until

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

            if label == "groq" and is_rate:
                _groq_cooldown_until = time.monotonic() + _COOLDOWN_SECS
                logger.warning(
                    f"[LLM] Groq rate-limited — cooling down {_COOLDOWN_SECS}s, "
                    f"falling back to Cerebras"
                )
            elif label == "cerebras" and is_rate:
                _cerebras_cooldown_until = time.monotonic() + _COOLDOWN_SECS
                logger.warning(
                    f"[LLM] Cerebras rate-limited — cooling down {_COOLDOWN_SECS}s, "
                    f"falling back to OpenRouter"
                )
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
