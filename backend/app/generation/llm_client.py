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
) -> str:
    """Generate a non-streaming response."""
    user_prompt = get_query_prompt(question, context, intent)

    if settings.llm_provider == "anthropic":
        return await _anthropic_generate(user_prompt)
    elif settings.llm_provider == "gemini":
        return await _gemini_generate(user_prompt)
    else:
        return await _openai_generate(user_prompt)


async def generate_response_stream(
    question: str,
    context: str,
    intent: str,
) -> AsyncGenerator[str, None]:
    """Generate a streaming response, yielding tokens as they arrive."""
    user_prompt = get_query_prompt(question, context, intent)

    if settings.llm_provider == "anthropic":
        async for token in _anthropic_stream(user_prompt):
            yield token
    elif settings.llm_provider == "gemini":
        async for token in _gemini_stream(user_prompt):
            yield token
    else:
        async for token in _openai_stream(user_prompt):
            yield token


async def _openai_generate(user_prompt: str) -> str:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    response = await client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=4096,
    )
    return response.choices[0].message.content


async def _openai_stream(user_prompt: str) -> AsyncGenerator[str, None]:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    stream = await client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=4096,
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


async def _anthropic_generate(user_prompt: str) -> str:
    from anthropic import AsyncAnthropic
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    response = await client.messages.create(
        model=settings.llm_model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text


async def _anthropic_stream(user_prompt: str) -> AsyncGenerator[str, None]:
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


async def _gemini_generate(user_prompt: str) -> str:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        api_key=settings.gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    response = await client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=4096,
    )
    return response.choices[0].message.content


async def _gemini_stream(user_prompt: str) -> AsyncGenerator[str, None]:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        api_key=settings.gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    stream = await client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=4096,
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content

