"""LLM client — unified interface for OpenAI and Anthropic with streaming support."""
from __future__ import annotations
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


def _get_gemini_model() -> str:
    """Ensure a valid Gemini model is used even if an old llama model was passed in env."""
    model = settings.llm_model
    if not model or "gemini" not in model.lower():
        return "gemini-1.5-flash"
    return model


async def _gemini_generate(user_prompt: str, history: list[dict] | None = None) -> str:
    from openai import AsyncOpenAI
    log.info(f"Generating Gemini response for prompt length: {len(user_prompt)}")
    if not settings.gemini_api_key:
        log.error("GEMINI_API_KEY is missing!")
        return "Error: Gemini API key is not configured."
        
    client = AsyncOpenAI(
        api_key=settings.gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})

    model = _get_gemini_model()
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=4096,
        )
        log.info("Gemini response generated successfully")
        return response.choices[0].message.content or ""
    except Exception as e:
        log.error(f"Gemini API error: {e}")
        raise


async def _gemini_stream(user_prompt: str, history: list[dict] | None = None) -> AsyncGenerator[str, None]:
    from openai import AsyncOpenAI
    log.info(f"Starting Gemini stream for prompt length: {len(user_prompt)}")
    if not settings.gemini_api_key:
        log.error("GEMINI_API_KEY is missing!")
        yield "Error: Gemini API key is not configured."
        return

    client = AsyncOpenAI(
        api_key=settings.gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})

    model = _get_gemini_model()
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=4096,
            stream=True,
        )
        log.info("Gemini stream connection established")
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
        log.info("Gemini stream complete")
    except Exception as e:
        log.error(f"Gemini streaming error: {e}")
        raise


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

