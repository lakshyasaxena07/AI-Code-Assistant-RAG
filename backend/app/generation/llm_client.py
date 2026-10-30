"""LLM client — unified interface for OpenAI and Anthropic with streaming support."""
from __future__ import annotations
import asyncio
from dataclasses import dataclass
import random
import re
import time
from typing import AsyncGenerator

from app.config import settings
from app.generation.prompts import SYSTEM_PROMPT, get_query_prompt
from app.utils.logger import get_logger

log = get_logger(__name__)


async def generate_response(
    question: str,
    context: str,
    intent: str,
    history: list[dict] | None = None,
) -> str:
    """Generate a non-streaming response."""
    user_prompt = get_query_prompt(question, context, intent)

    if settings.llm_provider == "anthropic":
        return await _anthropic_generate(user_prompt, history)
    elif settings.llm_provider == "gemini":
        return await _gemini_generate(user_prompt, history)
    elif settings.llm_provider == "groq":
        return await _groq_generate(user_prompt, history)
    else:
        return await _openai_generate(user_prompt, history)


async def generate_response_stream(
    question: str,
    context: str,
    intent: str,
    history: list[dict] | None = None,
) -> AsyncGenerator[str, None]:
    """Generate a streaming response, yielding tokens as they arrive."""
    user_prompt = get_query_prompt(question, context, intent)

    if settings.llm_provider == "anthropic":
        async for token in _anthropic_stream(user_prompt, history):
            yield token
    elif settings.llm_provider == "gemini":
        async for token in _gemini_stream(user_prompt, history):
            yield token
    elif settings.llm_provider == "groq":
        async for token in _groq_stream(user_prompt, history):
            yield token
    else:
        async for token in _openai_stream(user_prompt, history):
            yield token


async def _openai_generate(user_prompt: str, history: list[dict] | None = None) -> str:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})
    response = await client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=0.1,
        max_tokens=4096,
    )
    return response.choices[0].message.content or ""


async def _openai_stream(user_prompt: str, history: list[dict] | None = None) -> AsyncGenerator[str, None]:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})
    stream = await client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=0.1,
        max_tokens=4096,
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


async def _anthropic_generate(user_prompt: str, history: list[dict] | None = None) -> str:
    from anthropic import AsyncAnthropic
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    response = await client.messages.create(
        model=settings.llm_model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text


async def _anthropic_stream(user_prompt: str, history: list[dict] | None = None) -> AsyncGenerator[str, None]:
    from anthropic import AsyncAnthropic
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    async with client.messages.stream(
        model=settings.llm_model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    ) as stream:
        async for text in stream.text_stream:
            yield text


def _is_api_key_valid(key: str | None) -> bool:
    """Check if an API key is non-empty and not a placeholder."""
    if not key or not key.strip():
        return False
    normalized = key.strip().lower()
    placeholders = {
        "your_gemini_api_key_here",
        "your_gemini_api_key",
        "your_api_key_here",
        "your_api_key",
        "your-api-key",
        "placeholder",
    }
    if normalized in placeholders:
        return False
    if normalized.startswith("your_") or normalized.endswith("_here"):
        return False
    return True


class LLMError(RuntimeError):
    """Structured exception for LLM provider errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "upstream_error",
        status_code: int | None = None,
        retry_after: float | None = None,
        sanitized_details: str | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.retry_after = retry_after
        self.sanitized_details = sanitized_details

    def __str__(self) -> str:
        return self.message


@dataclass
class ErrorClassification:
    error_code: str
    status_code: int | None
    provider_status: str | None
    provider_code: str | None
    retry_after: float | None
    sanitized_message: str
    user_facing_message: str
    is_retryable: bool
    retry_delay: float | None


def sanitize_error_message(msg: str) -> str:
    """Sanitize any API keys or secrets from error messages."""
    if not msg:
        return ""
    for key_attr in ("gemini_api_key", "groq_api_key", "openai_api_key", "anthropic_api_key"):
        val = getattr(settings, key_attr, None)
        if val and isinstance(val, str) and len(val.strip()) > 4:
            msg = msg.replace(val.strip(), "[REDACTED_API_KEY]")
    # Redact Google AI Studio / Gemini API key format (AIza...)
    msg = re.sub(r"AIza[0-9A-Za-z_\-]{20,60}", "[REDACTED_KEY]", msg)
    # Redact key/token params
    msg = re.sub(r"(key|token|api_key|secret|password)=['\"]?([a-zA-Z0-9_\-]+)['\"]?", r"\1=[REDACTED]", msg, flags=re.IGNORECASE)
    msg = re.sub(r"Bearer\s+[a-zA-Z0-9_\-\.]+", "Bearer [REDACTED]", msg, flags=re.IGNORECASE)
    return msg


def _extract_retry_after(e: Exception) -> float | None:
    """Extract retry-after delay in seconds from exception or response headers."""
    direct = getattr(e, "retry_after", None)
    if isinstance(direct, (int, float)) and direct > 0:
        return float(direct)
    resp = getattr(e, "response", None)
    if resp is not None:
        headers = getattr(resp, "headers", None)
        if headers and hasattr(headers, "get"):
            header_val = headers.get("retry-after")
            if header_val:
                try:
                    val = float(header_val)
                    if val > 0:
                        return val
                except (ValueError, TypeError):
                    pass
    return None


def _format_gemini_error(e: Exception) -> str:
    """Extract clean provider error message, code, and status from SDK exception without leaking secrets."""
    body = getattr(e, "body", None)
    if isinstance(body, list) and len(body) > 0 and isinstance(body[0], dict):
        err = body[0].get("error", {})
        code = err.get("code", "")
        status = err.get("status", "")
        msg = err.get("message", "")
        parts = [p for p in [status, str(code) if code else ""] if p]
        prefix = f"Gemini API error ({' '.join(parts)})" if parts else "Gemini API error"
        raw = f"{prefix}: {msg}" if msg else prefix
        return sanitize_error_message(raw)
    elif isinstance(body, dict):
        err = body.get("error", {})
        if isinstance(err, dict):
            code = err.get("code", "")
            msg = err.get("message", "")
            prefix = f"Gemini API error ({code})" if code else "Gemini API error"
            raw = f"{prefix}: {msg}" if msg else prefix
            return sanitize_error_message(raw)
    return sanitize_error_message(str(e))


def _classify_gemini_error(e: Exception) -> ErrorClassification:
    """Analyze exception to determine HTTP status, provider error code, retryability, and user-facing message."""
    status_code = getattr(e, "status_code", None)
    if status_code is None:
        resp = getattr(e, "response", None)
        if resp is not None:
            status_code = getattr(resp, "status_code", None)

    body = getattr(e, "body", None)
    provider_code = None
    provider_status = None
    provider_msg = ""

    if isinstance(body, list) and len(body) > 0 and isinstance(body[0], dict):
        err = body[0].get("error", {})
        provider_code = str(err.get("code", "")) or None
        provider_status = err.get("status")
        provider_msg = err.get("message", "")
    elif isinstance(body, dict):
        err = body.get("error", {})
        if isinstance(err, dict):
            provider_code = str(err.get("code", "")) or None
            provider_status = err.get("status")
            provider_msg = err.get("message", "")

    if not status_code and provider_code and provider_code.isdigit():
        status_code = int(provider_code)

    retry_after = _extract_retry_after(e)
    raw_str = f"{provider_msg} {str(e)}".strip()
    sanitized_msg = sanitize_error_message(raw_str)
    msg_lower = sanitized_msg.lower()

    if status_code is None:
        m = re.search(r"\b([45]\d{2})\b", msg_lower)
        if m:
            status_code = int(m.group(1))

    # 1. Authentication & Permission Errors - NON-RETRYABLE
    is_auth = (
        status_code in (401, 403)
        or provider_status in ("UNAUTHENTICATED", "PERMISSION_DENIED")
        or (
            (status_code == 400 or provider_status == "INVALID_ARGUMENT")
            and any(k in msg_lower for k in ("api key", "api_key", "pass a valid", "invalid api key"))
        )
        or any(k in msg_lower for k in ("api key not valid", "pass a valid api key", "invalid api key", "unauthenticated"))
    )
    if is_auth:
        return ErrorClassification(
            error_code="auth_error",
            status_code=status_code or 401,
            provider_status=provider_status,
            provider_code=provider_code,
            retry_after=None,
            sanitized_message=sanitized_msg,
            user_facing_message=f"Gemini API authentication failed: {sanitized_msg}. Please verify your GEMINI_API_KEY in backend/.env.",
            is_retryable=False,
            retry_delay=None,
        )

    # 2. Quota Exhausted vs Rate Limit (429 or RESOURCE_EXHAUSTED)
    is_429 = (
        status_code == 429
        or provider_status == "RESOURCE_EXHAUSTED"
        or "429" in msg_lower
        or "resource_exhausted" in msg_lower
        or "rate limit" in msg_lower
        or "quota" in msg_lower
    )
    if is_429:
        is_quota = (
            "quota" in msg_lower
            or "queries per day" in msg_lower
            or "daily limit" in msg_lower
            or "billing" in msg_lower
            or retry_after is None
            or retry_after > 10.0
        )
        if is_quota:
            return ErrorClassification(
                error_code="quota_exhausted",
                status_code=429,
                provider_status=provider_status or "RESOURCE_EXHAUSTED",
                provider_code=provider_code,
                retry_after=retry_after,
                sanitized_message=sanitized_msg,
                user_facing_message=(
                    f"Gemini API quota exceeded ({sanitized_msg}). "
                    "Your account or key has reached its usage or billing limit. "
                    "Please check your quota in Google AI Studio or configure a new key."
                ),
                is_retryable=False,
                retry_delay=None,
            )
        else:
            return ErrorClassification(
                error_code="rate_limit",
                status_code=429,
                provider_status=provider_status or "RESOURCE_EXHAUSTED",
                provider_code=provider_code,
                retry_after=retry_after,
                sanitized_message=sanitized_msg,
                user_facing_message=(
                    f"Gemini API rate limit reached ({sanitized_msg}). "
                    "Please wait a moment before sending another query."
                ),
                is_retryable=True,
                retry_delay=retry_after,
            )

    # 3. 503 Overload / Service Unavailable - RETRYABLE
    is_503 = (
        status_code in (502, 503, 504)
        or provider_status in ("UNAVAILABLE", "SERVICE_UNAVAILABLE")
        or any(k in msg_lower for k in ("503", "unavailable", "overloaded", "bad gateway", "service unavailable"))
    )
    if is_503:
        return ErrorClassification(
            error_code="service_unavailable",
            status_code=status_code or 503,
            provider_status=provider_status or "UNAVAILABLE",
            provider_code=provider_code,
            retry_after=retry_after,
            sanitized_message=sanitized_msg,
            user_facing_message="The Gemini AI service is currently overloaded or temporarily unavailable (503 UNAVAILABLE). Please try again in a few moments.",
            is_retryable=True,
            retry_delay=None,
        )

    # 4. Timeout - RETRYABLE
    is_timeout = (
        isinstance(e, (asyncio.TimeoutError, TimeoutError))
        or "timeout" in msg_lower
        or "timed out" in msg_lower
    )
    if is_timeout:
        return ErrorClassification(
            error_code="timeout",
            status_code=504,
            provider_status=provider_status,
            provider_code=provider_code,
            retry_after=None,
            sanitized_message=sanitized_msg,
            user_facing_message="Request to Gemini timed out. The service took longer than expected to respond. Please try again.",
            is_retryable=True,
            retry_delay=None,
        )

    # 5. Connection Error - RETRYABLE
    is_conn = any(k in msg_lower for k in ("connection error", "failed to establish a new connection", "connecterror"))
    if is_conn:
        return ErrorClassification(
            error_code="connection_error",
            status_code=None,
            provider_status=provider_status,
            provider_code=provider_code,
            retry_after=None,
            sanitized_message=sanitized_msg,
            user_facing_message="Could not connect to Gemini API. Please check your network connection.",
            is_retryable=True,
            retry_delay=None,
        )

    # 6. Fallback Generic Upstream Error
    can_retry_generic = status_code is not None and status_code >= 500
    user_msg = _format_gemini_error(e)
    return ErrorClassification(
        error_code="upstream_error",
        status_code=status_code,
        provider_status=provider_status,
        provider_code=provider_code,
        retry_after=retry_after,
        sanitized_message=sanitized_msg,
        user_facing_message=user_msg,
        is_retryable=can_retry_generic,
        retry_delay=None,
    )


def _safe_float(val, default: float) -> float:
    return float(val) if isinstance(val, (int, float)) else default


def _safe_int(val, default: int) -> int:
    return int(val) if isinstance(val, int) and not isinstance(val, bool) else default


def _is_transient_error(e: Exception) -> bool:
    """Check if an exception is a transient error suitable for bounded retry."""
    return _classify_gemini_error(e).is_retryable


def _format_gemini_user_facing_error(e: Exception) -> str:
    """Extract clean, user-friendly error message without leaking internal tracebacks."""
    return _classify_gemini_error(e).user_facing_message


def _get_gemini_model() -> str:
    """Ensure a valid Gemini model is used even if an old llama model was passed in env."""
    model = settings.llm_model
    if not model or "gemini" not in model.lower():
        return "gemini-3.5-flash-lite"
    return model


async def _gemini_generate(user_prompt: str, history: list[dict] | None = None) -> str:
    from openai import AsyncOpenAI, Timeout
    log.info(f"Generating Gemini response for prompt length: {len(user_prompt)}")
    if not _is_api_key_valid(settings.gemini_api_key):
        log.error("GEMINI_API_KEY is not configured or is a placeholder.")
        return "Error: Gemini API key is not configured or is a placeholder. Please set a valid GEMINI_API_KEY in backend/.env."

    connect_timeout = _safe_float(getattr(settings, "llm_connect_timeout", 15.0), 15.0)
    read_timeout = _safe_float(getattr(settings, "llm_read_timeout", 60.0), 60.0)
    max_retries = _safe_int(getattr(settings, "llm_max_retries", 3), 3)
    retry_delay = _safe_float(getattr(settings, "llm_retry_delay", 1.0), 1.0)
    retry_jitter = _safe_float(getattr(settings, "llm_retry_jitter", 0.5), 0.5)

    timeout = Timeout(
        connect=connect_timeout,
        read=read_timeout,
        write=15.0,
        pool=15.0,
    )
    client = AsyncOpenAI(
        api_key=settings.gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        timeout=timeout,
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})

    model = _get_gemini_model()
    req_start_ts = time.strftime("%H:%M:%S")
    log.info(f"[TIMING] Gemini API request start: {req_start_ts} (model={model}, prompt_len={len(user_prompt)})")

    for attempt in range(max_retries + 1):
        try:
            t0 = time.perf_counter()
            async with asyncio.timeout(read_timeout):
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=4096,
                )
            t_call = time.perf_counter() - t0
            log.info(f"[TIMING] Gemini response received in {t_call:.2f}s (attempt {attempt + 1})")
            return response.choices[0].message.content or ""
        except Exception as e:
            info = _classify_gemini_error(e)
            if attempt < max_retries and info.is_retryable:
                if info.retry_delay is not None and info.retry_delay > 0:
                    backoff = info.retry_delay + random.uniform(0, retry_jitter)
                else:
                    backoff = min(10.0, retry_delay * (2 ** attempt)) + random.uniform(0, retry_jitter)
                log.warning(
                    f"Gemini transient error [{info.error_code}] on attempt {attempt + 1}/{max_retries + 1}: "
                    f"{info.sanitized_message}. Retrying in {backoff:.2f}s..."
                )
                await asyncio.sleep(backoff)
                continue
            log.error(f"Gemini API error [{info.error_code}, status={info.status_code}]: {info.user_facing_message}")
            raise LLMError(
                message=info.user_facing_message,
                error_code=info.error_code,
                status_code=info.status_code,
                retry_after=info.retry_after,
                sanitized_details=info.sanitized_message,
            ) from e


async def _gemini_stream(user_prompt: str, history: list[dict] | None = None) -> AsyncGenerator[str, None]:
    from openai import AsyncOpenAI, Timeout
    log.info(f"Starting Gemini stream for prompt length: {len(user_prompt)}")
    if not _is_api_key_valid(settings.gemini_api_key):
        log.error("GEMINI_API_KEY is not configured or is a placeholder.")
        yield "Error: Gemini API key is not configured or is a placeholder. Please set a valid GEMINI_API_KEY in backend/.env."
        return

    connect_timeout = _safe_float(getattr(settings, "llm_connect_timeout", 15.0), 15.0)
    read_timeout = _safe_float(getattr(settings, "llm_read_timeout", 60.0), 60.0)
    max_retries = _safe_int(getattr(settings, "llm_max_retries", 3), 3)
    retry_delay = _safe_float(getattr(settings, "llm_retry_delay", 1.0), 1.0)
    retry_jitter = _safe_float(getattr(settings, "llm_retry_jitter", 0.5), 0.5)

    timeout = Timeout(
        connect=connect_timeout,
        read=read_timeout,
        write=15.0,
        pool=15.0,
    )
    client = AsyncOpenAI(
        api_key=settings.gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        timeout=timeout,
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})

    model = _get_gemini_model()
    req_start_ts = time.strftime("%H:%M:%S")
    log.info(f"[TIMING] Gemini API request start: {req_start_ts} (model={model}, prompt_len={len(user_prompt)})")

    start_time = time.perf_counter()
    yielded_any = False

    for attempt in range(max_retries + 1):
        try:
            t_conn_start = time.perf_counter()
            async with asyncio.timeout(connect_timeout):
                stream = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=4096,
                    stream=True,
                )
            t_connected = time.perf_counter() - t_conn_start
            log.info(f"[TIMING] Gemini stream connection established in {t_connected:.2f}s (attempt {attempt + 1})")

            t_first_token = None
            total_tokens = 0
            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    if not yielded_any:
                        yielded_any = True
                        t_first_token = time.perf_counter() - start_time
                        log.info(f"[TIMING] Gemini first token received in {t_first_token:.2f}s (TTFT)")
                    total_tokens += 1
                    yield delta.content

            total_duration = time.perf_counter() - start_time
            log.info(f"[TIMING] Gemini streaming complete in {total_duration:.2f}s ({total_tokens} tokens)")
            return

        except Exception as e:
            info = _classify_gemini_error(e)
            # RULE: If tokens were already yielded, DO NOT retry from beginning (prevents duplicate output)
            if yielded_any:
                log.error(f"Gemini stream interrupted after partial output: {info.user_facing_message}")
                yield f"\n\n[Generation interrupted: {info.user_facing_message}]"
                return

            if attempt < max_retries and info.is_retryable:
                if info.retry_delay is not None and info.retry_delay > 0:
                    backoff = info.retry_delay + random.uniform(0, retry_jitter)
                else:
                    backoff = min(10.0, retry_delay * (2 ** attempt)) + random.uniform(0, retry_jitter)
                log.warning(
                    f"Gemini transient error [{info.error_code}] on attempt {attempt + 1}/{max_retries + 1}: "
                    f"{info.sanitized_message}. Retrying in {backoff:.2f}s..."
                )
                await asyncio.sleep(backoff)
                continue

            log.error(
                f"Gemini streaming failed (attempt {attempt + 1}/{max_retries + 1}) "
                f"[{info.error_code}, status={info.status_code}]: {info.user_facing_message}"
            )
            raise LLMError(
                message=info.user_facing_message,
                error_code=info.error_code,
                status_code=info.status_code,
                retry_after=info.retry_after,
                sanitized_details=info.sanitized_message,
            ) from e


async def _groq_generate(user_prompt: str, history: list[dict] | None = None) -> str:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        api_key=settings.groq_api_key,
        base_url="https://api.groq.com/openai/v1"
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})

    response = await client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=0.1,
        max_tokens=4096,
    )
    return response.choices[0].message.content


async def _groq_stream(user_prompt: str, history: list[dict] | None = None) -> AsyncGenerator[str, None]:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        api_key=settings.groq_api_key,
        base_url="https://api.groq.com/openai/v1"
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})

    stream = await client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=0.1,
        max_tokens=4096,
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content

