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
    _is_api_key_valid,
    _format_gemini_error,
    _classify_gemini_error,
    sanitize_error_message,
    LLMError,
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
            mock_settings.llm_max_retries = 1
            mock_settings.llm_retry_delay = 0.001
            mock_settings.llm_retry_jitter = 0.0

            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("Stream connection error"))

            with patch("openai.AsyncOpenAI", return_value=mock_client):
                with pytest.raises(RuntimeError, match="Could not connect to Gemini API"):
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


# ── 5. Regression Tests for HTTP 400 & Placeholder Keys ───────────────────────

def test_is_api_key_valid_detection():
    """Verify placeholder and invalid keys are detected correctly."""
    assert not _is_api_key_valid("")
    assert not _is_api_key_valid("   ")
    assert not _is_api_key_valid("your_gemini_api_key_here")
    assert not _is_api_key_valid("your_gemini_api_key")
    assert not _is_api_key_valid("your_api_key_here")
    assert not _is_api_key_valid("placeholder")
    assert _is_api_key_valid("AIzaSyD-dummy-valid-looking-key")
    assert _is_api_key_valid("test-key")


def test_gemini_placeholder_key_in_generate_and_stream():
    """Verify placeholder key returns/yields clean error without making network calls."""
    async def _test():
        with patch("app.generation.llm_client.settings") as mock_settings:
            mock_settings.gemini_api_key = "your_gemini_api_key_here"

            # Generate non-streaming
            res = await _gemini_generate("Hello", history=None)
            assert "Error: Gemini API key is not configured or is a placeholder" in res

            # Generate streaming
            tokens = [t async for t in _gemini_stream("Hello", history=None)]
            assert len(tokens) == 1
            assert "Error: Gemini API key is not configured or is a placeholder" in tokens[0]
    asyncio.run(_test())


def test_gemini_http_400_error_formatting():
    """Verify HTTP 400 error body from Gemini OpenAI endpoint is cleanly extracted."""
    # Gemini OpenAI endpoint returns a list with an error object
    mock_body = [{
        "error": {
            "code": 400,
            "message": "Please pass a valid API key",
            "status": "INVALID_ARGUMENT",
        }
    }]
    err = Exception("Error code: 400")
    err.body = mock_body

    formatted = _format_gemini_error(err)
    assert formatted == "Gemini API error (INVALID_ARGUMENT 400): Please pass a valid API key"
    assert "Bearer" not in formatted



# ── 6. Timeout, 503 Retry & Streaming Failure Tests ─────────────────────────

def test_gemini_stream_503_retry_and_recover():
    """Verify _gemini_stream retries on transient 503 and succeeds on next attempt."""
    async def _test():
        with patch("app.generation.llm_client.settings") as mock_settings:
            mock_settings.gemini_api_key = "test-key"
            mock_settings.llm_model = "gemini-1.5-flash"
            mock_settings.llm_max_retries = 2
            mock_settings.llm_retry_delay = 0.01
            mock_settings.llm_retry_jitter = 0.0

            async def mock_chunks():
                chunk = MagicMock()
                chunk.choices = [MagicMock(delta=MagicMock(content="Success after 503"))]
                yield chunk

            err_503 = Exception("503 UNAVAILABLE: The model is overloaded. Please try again later.")
            mock_client = MagicMock()
            # Attempt 1: 503, Attempt 2: success
            mock_client.chat.completions.create = AsyncMock(
                side_effect=[err_503, mock_chunks()]
            )

            with patch("openai.AsyncOpenAI", return_value=mock_client):
                tokens = []
                async for tok in _gemini_stream("Test 503 recovery", history=None):
                    tokens.append(tok)

                assert "".join(tokens) == "Success after 503"
                assert mock_client.chat.completions.create.await_count == 2
    asyncio.run(_test())


def test_gemini_stream_503_max_retries_exceeded():
    """Verify _gemini_stream exhausts bounded retries and raises user-friendly 503 message."""
    async def _test():
        with patch("app.generation.llm_client.settings") as mock_settings:
            mock_settings.gemini_api_key = "test-key"
            mock_settings.llm_model = "gemini-1.5-flash"
            mock_settings.llm_max_retries = 2
            mock_settings.llm_retry_delay = 0.01
            mock_settings.llm_retry_jitter = 0.0

            err_503 = Exception("503 Service Unavailable")
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(side_effect=err_503)

            with patch("openai.AsyncOpenAI", return_value=mock_client):
                with pytest.raises(RuntimeError, match="503 UNAVAILABLE"):
                    async for _ in _gemini_stream("Test 503 max retries", history=None):
                        pass

                # Initial attempt + 2 retries = 3 total attempts
                assert mock_client.chat.completions.create.await_count == 3
    asyncio.run(_test())


def test_gemini_stream_partial_failure_does_not_retry_or_duplicate():
    """Verify partial failure during streaming does NOT retry from start (avoids duplicate tokens)."""
    async def _test():
        with patch("app.generation.llm_client.settings") as mock_settings:
            mock_settings.gemini_api_key = "test-key"
            mock_settings.llm_model = "gemini-1.5-flash"
            mock_settings.llm_max_retries = 3

            async def mock_failing_stream():
                # Yield first token
                c1 = MagicMock()
                c1.choices = [MagicMock(delta=MagicMock(content="Partial output"))]
                yield c1
                # Then fail mid-stream
                raise RuntimeError("Connection dropped mid-stream")

            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_failing_stream())

            with patch("openai.AsyncOpenAI", return_value=mock_client):
                tokens = []
                async for tok in _gemini_stream("Test mid-stream drop", history=None):
                    tokens.append(tok)

                full = "".join(tokens)
                assert "Partial output" in full
                assert "[Generation interrupted:" in full
                # Must NOT call create a second time (no retry after tokens yielded)
                assert mock_client.chat.completions.create.await_count == 1
    asyncio.run(_test())


def test_gemini_timeout_user_facing_error():
    """Verify TimeoutError raises a clear user-facing message."""
    async def _test():
        with patch("app.generation.llm_client.settings") as mock_settings:
            mock_settings.gemini_api_key = "test-key"
            mock_settings.llm_model = "gemini-1.5-flash"
            mock_settings.llm_max_retries = 1
            mock_settings.llm_retry_delay = 0.01
            mock_settings.llm_retry_jitter = 0.0

            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(side_effect=TimeoutError("Request timed out"))

            with patch("openai.AsyncOpenAI", return_value=mock_client):
                with pytest.raises(RuntimeError, match="Request to Gemini timed out"):
                    await _gemini_generate("Test timeout", history=None)
    asyncio.run(_test())


def test_index_caching_and_invalidation():
    """Verify load_bm25 and load_vector_store cache results and clear_indexer_cache purges them."""
    from app.ingestion.indexer import load_bm25, load_vector_store, clear_indexer_cache, _bm25_cache, _vector_cache

    clear_indexer_cache()
    assert len(_bm25_cache) == 0
    assert len(_vector_cache) == 0

    # Mock file existence and loading
    mock_bm25_data = {"bm25": MagicMock(), "chunk_ids": ["c1"], "chunks_data": [{"chunk_id": "c1"}]}
    mock_vec_data = {"ids": ["c1"], "embeddings": [[0.1, 0.2]]}

    with patch("pathlib.Path.exists", return_value=True), patch("builtins.open", MagicMock()), patch("pickle.load", side_effect=[mock_bm25_data, mock_vec_data]):
        res1 = load_bm25("repo123")
        assert res1 is not None
        assert "repo123" in _bm25_cache

        # Second load should come from cache without calling pickle.load again
        res2 = load_bm25("repo123")
        assert res2 is res1

        v1 = load_vector_store("repo123")
        assert v1 is not None
        assert "repo123" in _vector_cache

        v2 = load_vector_store("repo123")
        assert v2 is v1

        # Clear specific repo
        clear_indexer_cache("repo123")
        assert "repo123" not in _bm25_cache
        assert "repo123" not in _vector_cache


# ── 7. Regression Tests for 429 Quota, Rate Limit, Auth, and Secret Redaction ───

def test_gemini_429_quota_exhausted_zero_retries():
    """Verify HTTP 429 quota exhaustion raises quota_exhausted and does NOT retry."""
    async def _test():
        with patch("app.generation.llm_client.settings") as mock_settings:
            mock_settings.gemini_api_key = "test-key"
            mock_settings.llm_model = "gemini-1.5-flash"
            mock_settings.llm_max_retries = 3
            mock_settings.llm_retry_delay = 0.001
            mock_settings.llm_retry_jitter = 0.0

            err_429 = Exception("429 RESOURCE_EXHAUSTED: Resource has been exhausted (e.g. check quota).")
            err_429.status_code = 429
            err_429.body = {
                "error": {
                    "code": 429,
                    "message": "Resource has been exhausted (e.g. check quota).",
                    "status": "RESOURCE_EXHAUSTED",
                }
            }

            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(side_effect=err_429)

            with patch("openai.AsyncOpenAI", return_value=mock_client):
                with pytest.raises(LLMError) as exc_info:
                    async for _ in _gemini_stream("Test 429 quota", history=None):
                        pass

                # Assert non-retryable: only 1 attempt must be made
                assert mock_client.chat.completions.create.await_count == 1
                assert exc_info.value.error_code == "quota_exhausted"
                assert exc_info.value.status_code == 429
                assert "quota" in exc_info.value.message.lower()
    asyncio.run(_test())


def test_gemini_429_rate_limit_with_retry_after_honored():
    """Verify 429 with short Retry-After delays and retries successfully."""
    async def _test():
        with patch("app.generation.llm_client.settings") as mock_settings:
            mock_settings.gemini_api_key = "test-key"
            mock_settings.llm_model = "gemini-1.5-flash"
            mock_settings.llm_max_retries = 2
            mock_settings.llm_retry_jitter = 0.0

            err_429 = Exception("429 Too Many Requests: Rate limit exceeded")
            err_429.status_code = 429
            err_429.retry_after = 0.02
            mock_resp = MagicMock()
            mock_resp.headers = {"retry-after": "0.02"}
            err_429.response = mock_resp

            async def mock_chunks():
                chunk = MagicMock()
                chunk.choices = [MagicMock(delta=MagicMock(content="Recovered after rate limit"))]
                yield chunk

            mock_client = MagicMock()
            # Attempt 1: 429 with short retry-after, Attempt 2: success
            mock_client.chat.completions.create = AsyncMock(side_effect=[err_429, mock_chunks()])

            with patch("openai.AsyncOpenAI", return_value=mock_client):
                tokens = []
                async for tok in _gemini_stream("Test rate limit recovery", history=None):
                    tokens.append(tok)

                assert "".join(tokens) == "Recovered after rate limit"
                assert mock_client.chat.completions.create.await_count == 2
    asyncio.run(_test())


def test_gemini_auth_failure_zero_retries():
    """Verify 401/403 or invalid API key raises auth_error and does NOT retry."""
    async def _test():
        with patch("app.generation.llm_client.settings") as mock_settings:
            mock_settings.gemini_api_key = "test-key"
            mock_settings.llm_model = "gemini-1.5-flash"
            mock_settings.llm_max_retries = 3

            err_401 = Exception("401 Unauthorized: API key not valid. Please pass a valid API key.")
            err_401.status_code = 401
            err_401.body = [{"error": {"code": 401, "message": "API key not valid", "status": "UNAUTHENTICATED"}}]

            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(side_effect=err_401)

            with patch("openai.AsyncOpenAI", return_value=mock_client):
                with pytest.raises(LLMError) as exc_info:
                    await _gemini_generate("Test auth failure", history=None)

                # Assert non-retryable: exactly 1 attempt
                assert mock_client.chat.completions.create.await_count == 1
                assert exc_info.value.error_code == "auth_error"
                assert exc_info.value.status_code == 401
                assert "authentication failed" in exc_info.value.message.lower()
    asyncio.run(_test())


def test_gemini_503_overload_retries_and_structured_error():
    """Verify 503 overload retries up to max_retries and sets error_code=service_unavailable."""
    async def _test():
        with patch("app.generation.llm_client.settings") as mock_settings:
            mock_settings.gemini_api_key = "test-key"
            mock_settings.llm_model = "gemini-1.5-flash"
            mock_settings.llm_max_retries = 2
            mock_settings.llm_retry_delay = 0.001
            mock_settings.llm_retry_jitter = 0.0

            err_503 = Exception("503 UNAVAILABLE: The model is overloaded. Please try again later.")
            err_503.status_code = 503

            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(side_effect=err_503)

            with patch("openai.AsyncOpenAI", return_value=mock_client):
                with pytest.raises(LLMError) as exc_info:
                    async for _ in _gemini_stream("Test 503 overload", history=None):
                        pass

                # Initial attempt + 2 retries = 3
                assert mock_client.chat.completions.create.await_count == 3
                assert exc_info.value.error_code == "service_unavailable"
                assert exc_info.value.status_code == 503
    asyncio.run(_test())


def test_gemini_timeout_retries_and_structured_error():
    """Verify TimeoutError retries up to max_retries and sets error_code=timeout."""
    async def _test():
        with patch("app.generation.llm_client.settings") as mock_settings:
            mock_settings.gemini_api_key = "test-key"
            mock_settings.llm_model = "gemini-1.5-flash"
            mock_settings.llm_max_retries = 2
            mock_settings.llm_retry_delay = 0.001
            mock_settings.llm_retry_jitter = 0.0

            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(side_effect=asyncio.TimeoutError())

            with patch("openai.AsyncOpenAI", return_value=mock_client):
                with pytest.raises(LLMError) as exc_info:
                    await _gemini_generate("Test timeout error", history=None)

                # Initial attempt + 2 retries = 3
                assert mock_client.chat.completions.create.await_count == 3
                assert exc_info.value.error_code == "timeout"
                assert exc_info.value.status_code == 504
    asyncio.run(_test())


def test_sensitive_data_redaction():
    """Verify secrets and keys are scrubbed and never appear in error messages."""
    secret_key = "AIzaSyD-dummy-secret-gemini-key-12345"
    with patch("app.generation.llm_client.settings") as mock_settings:
        mock_settings.gemini_api_key = secret_key
        mock_settings.groq_api_key = "gsk_test_groq_secret_67890"

        # Raw message containing configured key, unknown AIza key, and generic token
        raw = (
            f"Error at https://generativelanguage.googleapis.com?key={secret_key} "
            f"and secondary=AIzaSyB9876543210123456789012345678901 "
            f"token=secret_jwt_token_here with Bearer {secret_key}"
        )
        sanitized = sanitize_error_message(raw)

        assert secret_key not in sanitized
        assert "AIzaSyB" not in sanitized
        assert "secret_jwt_token_here" not in sanitized
        assert "[REDACTED" in sanitized


def test_websocket_and_http_query_error_propagation():
    """Verify LLMError translates to structured error packet on WebSocket and HTTP status on REST."""
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)

    # 1. Test WebSocket error packet
    with patch("app.api.routes_query.hybrid_retrieve", return_value=[]), \
         patch("app.api.routes_query.rerank", return_value=[]), \
         patch("app.api.routes_query.graph_expand", return_value=[]), \
         patch("app.api.routes_query.build_context", return_value=("ctx", [])):

        async def mock_err_stream(*args, **kwargs):
            raise LLMError(
                message="Gemini API quota exceeded.",
                error_code="quota_exhausted",
                status_code=429,
                retry_after=None,
            )
            yield "token"

        with patch("app.api.routes_query.generate_response_stream", side_effect=mock_err_stream):
            with client.websocket_connect("/api/query/stream") as ws:
                ws.send_json({"repo_id": "test_repo", "question": "test question"})
                msg = ws.receive_json()
                if msg.get("type") == "retrieval_info":
                    msg = ws.receive_json()
                assert msg.get("type") == "error"
                assert msg.get("error_code") == "quota_exhausted"
                assert msg.get("status_code") == 429
                assert "quota exceeded" in msg.get("data", "").lower()

    # 2. Test HTTP REST endpoint status translation
    with patch("app.api.routes_query.hybrid_retrieve", return_value=[]), \
         patch("app.api.routes_query.rerank", return_value=[]), \
         patch("app.api.routes_query.graph_expand", return_value=[]), \
         patch("app.api.routes_query.build_context", return_value=("ctx", [])):

        with patch("app.api.routes_query.generate_response", side_effect=LLMError("Auth failed", error_code="auth_error", status_code=401)):
            res = client.post("/api/query/", json={"repo_id": "test_repo", "question": "test"})
            assert res.status_code == 401
            assert res.json()["detail"]["error_code"] == "auth_error"
