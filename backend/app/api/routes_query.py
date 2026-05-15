"""Query API routes — handles chat queries with the full RAG pipeline."""
from __future__ import annotations
import json
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.models.schemas import QueryRequest, QueryResponse, RetrievalInfo, RetrievedChunk
from app.retrieval.query_analyzer import classify_intent, rewrite_query, get_intent_config
from app.retrieval.hybrid_retriever import hybrid_retrieve
from app.retrieval.reranker import rerank
from app.retrieval.graph_retriever import graph_expand
from app.retrieval.context_builder import build_context
from app.generation.llm_client import generate_response, generate_response_stream
from app.generation.memory import get_or_create_conversation
from app.generation.guard import verify_response
from app.utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/api/query", tags=["query"])


@router.post("/", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Full RAG query pipeline — non-streaming."""
    result = await _execute_pipeline(
        repo_id=request.repo_id,
        question=request.question,
        conversation_id=request.conversation_id,
    )
    return result


@router.websocket("/stream")
async def query_stream(websocket: WebSocket):
    """WebSocket endpoint for streaming responses.

    Client sends: {"repo_id": "...", "question": "...", "conversation_id": "..."}
    Server streams: {"type": "token", "data": "..."} for each token
                    {"type": "retrieval_info", "data": {...}} with retrieval metadata
                    {"type": "done"} when complete
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            request = json.loads(data)

            repo_id = request.get("repo_id", "")
            question = request.get("question", "")
            conversation_id = request.get("conversation_id")

            if not repo_id or not question:
                await websocket.send_json({"type": "error", "data": "Missing repo_id or question"})
                continue

            # Execute retrieval pipeline
            start = time.time()
            conversation = get_or_create_conversation(conversation_id)
            context_summary = conversation.get_context_summary()

            intent = classify_intent(question)
            intent_config = get_intent_config(intent)
            rewritten = rewrite_query(question, intent, context_summary)

            top_k = int(15 * intent_config["top_k_multiplier"])
            candidates = hybrid_retrieve(rewritten, repo_id, top_k=top_k)

            reranked = rerank(question, candidates)

            hops = intent_config["graph_hops"]
            expanded = graph_expand(reranked, repo_id, hops=hops)

            context, used_chunks = build_context(expanded)

            retrieval_ms = (time.time() - start) * 1000

            # Send retrieval info
            retrieval_info = _build_retrieval_info(
                question, rewritten, intent,
                len(candidates), len(candidates), len(reranked),
                retrieval_ms, used_chunks,
            )
            await websocket.send_json({
                "type": "retrieval_info",
                "data": retrieval_info.model_dump(),
            })

            # Stream LLM response
            full_response = ""
            async for token in generate_response_stream(question, context, intent):
                full_response += token
                await websocket.send_json({"type": "token", "data": token})

            # Verify response
            verification = verify_response(full_response, used_chunks)

            # Update conversation memory
            referenced_files = list({
                c.get("metadata", {}).get("file_path", "")
                for c in used_chunks
            })
            conversation.add_turn("user", question)
            conversation.add_turn("assistant", full_response[:500], referenced_files)

            await websocket.send_json({
                "type": "done",
                "data": {
                    "conversation_id": conversation.conversation_id,
                    "verification": verification,
                },
            })

    except WebSocketDisconnect:
        log.info("WebSocket client disconnected")
    except Exception as e:
        log.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "data": str(e)})
        except Exception:
            pass


async def _execute_pipeline(
    repo_id: str,
    question: str,
    conversation_id: str | None = None,
) -> QueryResponse:
    """Execute the full RAG pipeline."""
    start = time.time()

    conversation = get_or_create_conversation(conversation_id)
    context_summary = conversation.get_context_summary()

    # 1. Classify intent
    intent = classify_intent(question)
    intent_config = get_intent_config(intent)

    # 2. Rewrite query
    rewritten = rewrite_query(question, intent, context_summary)

    # 3. Hybrid retrieval
    top_k = int(15 * intent_config["top_k_multiplier"])
    candidates = hybrid_retrieve(rewritten, repo_id, top_k=top_k)

    # 4. Rerank
    reranked = rerank(question, candidates)

    # 5. Graph expansion
    hops = intent_config["graph_hops"]
    expanded = graph_expand(reranked, repo_id, hops=hops)

    # 6. Build context
    context, used_chunks = build_context(expanded)

    retrieval_ms = (time.time() - start) * 1000

    # 7. Generate response
    answer = await generate_response(question, context, intent)

    # 8. Verify
    verify_response(answer, used_chunks)

    # 9. Update conversation
    referenced_files = list({
        c.get("metadata", {}).get("file_path", "")
        for c in used_chunks
    })
    conversation.add_turn("user", question)
    conversation.add_turn("assistant", answer[:500], referenced_files)

    retrieval_info = _build_retrieval_info(
        question, rewritten, intent,
        len(candidates), len(candidates), len(reranked),
        retrieval_ms, used_chunks,
    )

    return QueryResponse(
        answer=answer,
        retrieval_info=retrieval_info,
        conversation_id=conversation.conversation_id,
    )


def _build_retrieval_info(
    original: str, rewritten: str, intent: str,
    bm25_count: int, vector_count: int, reranked_count: int,
    latency_ms: float, used_chunks: list[dict],
) -> RetrievalInfo:
    chunks = []
    for c in used_chunks:
        meta = c.get("metadata", {})
        chunks.append(RetrievedChunk(
            chunk_id=c.get("chunk_id", ""),
            file_path=meta.get("file_path", ""),
            name=meta.get("name", ""),
            chunk_type=meta.get("chunk_type", ""),
            start_line=meta.get("start_line", 0) if isinstance(meta.get("start_line"), int) else 0,
            end_line=meta.get("end_line", 0) if isinstance(meta.get("end_line"), int) else 0,
            content=c.get("content", "")[:500],  # Truncate for API response
            score=round(c.get("rerank_score", c.get("rrf_score", 0)), 4),
            language=meta.get("language", ""),
        ))

    return RetrievalInfo(
        original_query=original,
        rewritten_query=rewritten,
        intent=intent,
        bm25_count=bm25_count,
        vector_count=vector_count,
        reranked_count=reranked_count,
        latency_ms=round(latency_ms, 1),
        retrieved_chunks=chunks,
    )
