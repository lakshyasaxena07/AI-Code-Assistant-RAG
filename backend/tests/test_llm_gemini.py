"""Tests for Gemini LLM provider integration, configuration, and error handling."""
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.config import Settings
from app.generation.llm_client import (
    generate_response,
    generate_response_stream,
    _gemini_generate,
    _gemini_stream,
    _get_gemini_model,
)


# ── 1. Provider Configuration Tests ──────────────────────────────────────────

def test_default_gemini_configuration():
    """Verify default provider and model are set to Gemini free tier."""
    s = Settings(
        _env_file=None,
        openai_api_key="",
        anthropic_api_key="",
        gemini_api_key="",
        groq_api_key="",
    )
    assert s.llm_provider == "gemini"
    assert s.llm_model == "gemini-1.5-flash"


def test_gemini_api_key_env_var_and_alias():
    """Verify GEMINI_API_KEY and GOOGLE_API_KEY alias load correctly."""
    # Test GEMINI_API_KEY
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-gemini-secret-key"}):
        s = Settings(_env_file=None)
        assert s.gemini_api_key == "test-gemini-secret-key"

    # Test GOOGLE_API_KEY fallback alias
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-google-secret-key"}, clear=True):
        s = Settings(_env_file=None)
        assert s.gemini_api_key == "test-google-secret-key"


def test_huggingface_env_override():
    """Verify HF Space environment variables override local defaults."""
    env = {
        "LLM_PROVIDER": "gemini",
        "LLM_MODEL": "gemini-2.0-flash",
        "GEMINI_API_KEY": "hf-secret-key-12345",
    }
    with patch.dict(os.environ, env):
        s = Settings(_env_file=None)
        assert s.llm_provider == "gemini"
        assert s.llm_model == "gemini-2.0-flash"
        assert s.gemini_api_key == "hf-secret-key-12345"


def test_gemini_model_fallback():
    """Verify _get_gemini_model falls back to gemini-1.5-flash if an invalid non-gemini model is set."""
    with patch("app.generation.llm_client.settings") as mock_settings:
        mock_settings.llm_model = "llama-3.3-70b-versatile"
        assert _get_gemini_model() == "gemini-1.5-flash"

        mock_settings.llm_model = ""
        assert _get_gemini_model() == "gemini-1.5-flash"

        mock_settings.llm_model = "gemini-2.0-flash"
        assert _get_gemini_model() == "gemini-2.0-flash"


import asyncio


# ── 2. Missing API Key Tests ──────────────────────────────────────────────────

def test_gemini_generate_missing_key():
    """Verify _gemini_generate returns a descriptive error when API key is missing."""
    async def _test():
        with patch("app.generation.llm_client.settings") as mock_settings:
            mock_settings.gemini_api_key = ""
            result = await _gemini_generate("Hello", history=None)
            assert "Error: Gemini API key is not configured" in result
    asyncio.run(_test())


def test_gemini_stream_missing_key():
    """Verify _gemini_stream yields a descriptive error when API key is missing."""
    async def _test():
        with patch("app.generation.llm_client.settings") as mock_settings:
            mock_settings.gemini_api_key = ""
            tokens = [token async for token in _gemini_stream("Hello", history=None)]
            assert len(tokens) == 1
            assert "Error: Gemini API key is not configured" in tokens[0]
    asyncio.run(_test())


# ── 3. API Error Handling Tests ──────────────────────────────────────────────

def test_gemini_generate_api_error():
    """Verify _gemini_generate logs and raises on API exceptions."""
    async def _test():
        with patch("app.generation.llm_client.settings") as mock_settings:
            mock_settings.gemini_api_key = "test-key"
            mock_settings.llm_model = "gemini-1.5-flash"

            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("Quota exceeded or API down"))

            with patch("openai.AsyncOpenAI", return_value=mock_client):
                with pytest.raises(RuntimeError, match="Quota exceeded or API down"):
                    await _gemini_generate("Analyze code", history=None)
    asyncio.run(_test())


def test_gemini_stream_api_error():
    """Verify _gemini_stream propagates exceptions when API fails."""
    async def _test():
        with patch("app.generation.llm_client.settings") as mock_settings:
            mock_settings.gemini_api_key = "test-key"
            mock_settings.llm_model = "gemini-1.5-flash"

            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("Stream connection error"))

            with patch("openai.AsyncOpenAI", return_value=mock_client):
                with pytest.raises(RuntimeError, match="Stream connection error"):
                    async for _ in _gemini_stream("Analyze code", history=None):
                        pass
    asyncio.run(_test())


# ── 4. Successful Response Handling Tests ────────────────────────────────────

def test_gemini_generate_success():
    """Verify _gemini_generate parses successful response with prompt and history."""
    async def _test():
        with patch("app.generation.llm_client.settings") as mock_settings:
            mock_settings.gemini_api_key = "test-key"
            mock_settings.llm_model = "gemini-1.5-flash"

            # Mock OpenAI chat completion response
            mock_choice = MagicMock()
            mock_choice.message.content = "Here is the explanation of the function [main.py:10-25]."
            mock_response = MagicMock(choices=[mock_choice])

            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            with patch("openai.AsyncOpenAI", return_value=mock_client):
                history = [{"role": "user", "content": "Previous question"}]
                answer = await _gemini_generate("What does this function do?", history=history)

                assert answer == "Here is the explanation of the function [main.py:10-25]."
                mock_client.chat.completions.create.assert_awaited_once()

                call_kwargs = mock_client.chat.completions.create.await_args.kwargs
                assert call_kwargs["model"] == "gemini-1.5-flash"
                assert len(call_kwargs["messages"]) == 3  # system, history user, current user
    asyncio.run(_test())


def test_gemini_stream_success():
    """Verify _gemini_stream yields delta tokens correctly."""
    async def _test():
        with patch("app.generation.llm_client.settings") as mock_settings:
            mock_settings.gemini_api_key = "test-key"
            mock_settings.llm_model = "gemini-1.5-flash"

            async def mock_chunks():
                tokens = ["Function", " ", "calculates", " ", "hash", " [hash.py:5-10]"]
                for t in tokens:
                    chunk = MagicMock()
                    chunk.choices = [MagicMock(delta=MagicMock(content=t))]
                    yield chunk

            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_chunks())

            with patch("openai.AsyncOpenAI", return_value=mock_client):
                streamed = []
                async for token in _gemini_stream("Explain hash function", history=None):
                    streamed.append(token)

                assert "".join(streamed) == "Function calculates hash [hash.py:5-10]"
    asyncio.run(_test())


def test_generate_response_routing_gemini():
    """Verify top-level generate_response and generate_response_stream route to Gemini."""
    async def _test():
        with patch("app.generation.llm_client.settings") as mock_settings:
            mock_settings.llm_provider = "gemini"
            mock_settings.gemini_api_key = "test-key"
            mock_settings.llm_model = "gemini-1.5-flash"

            with patch("app.generation.llm_client._gemini_generate", new_callable=AsyncMock) as mock_gen:
                mock_gen.return_value = "Mocked Gemini Answer"
                ans = await generate_response("query", "code context", "explain", history=None)
                assert ans == "Mocked Gemini Answer"
                mock_gen.assert_awaited_once()

            async def mock_stream_gen(user_prompt, history):
                for tok in ["token1", "token2"]:
                    yield tok

            with patch("app.generation.llm_client._gemini_stream", side_effect=mock_stream_gen):
                tokens = [t async for t in generate_response_stream("query", "code context", "explain")]
                assert tokens == ["token1", "token2"]
    asyncio.run(_test())
