"""Conversation memory with query rewriting.

Instead of stuffing full conversation history into the retrieval query
(which adds noise and reduces precision), we track context and rewrite
follow-up queries to be self-contained.

"what about the other function?" → "explain the refresh_token function in auth/jwt.py"
"""
from __future__ import annotations
from dataclasses import dataclass, field
import uuid


@dataclass
class ConversationTurn:
    role: str  # "user" or "assistant"
    content: str
    referenced_files: list[str] = field(default_factory=list)


@dataclass
class Conversation:
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    turns: list[ConversationTurn] = field(default_factory=list)
    referenced_files: set[str] = field(default_factory=set)

    def add_turn(self, role: str, content: str, files: list[str] | None = None):
        self.turns.append(ConversationTurn(role=role, content=content,
                                           referenced_files=files or []))
        if files:
            self.referenced_files.update(files)

    def get_history(self, max_turns: int = 5) -> list[dict]:
        """Return full history in OpenAI message format for the LLM."""
        recent = self.turns[-max_turns * 2:]
        return [{"role": t.role, "content": t.content} for t in recent]

    def get_context_summary(self, max_turns: int = 3) -> str:
        """Build a concise summary of recent conversation for query rewriting."""
        if not self.turns:
            return ""

        recent = self.turns[-max_turns * 2:]  # Last N Q&A pairs
        parts = []
        for turn in recent:
            prefix = "User asked" if turn.role == "user" else "Assistant discussed"
            # Only include first 150 chars to keep it concise
            summary = turn.content[:150]
            parts.append(f"{prefix}: {summary}")

        context = "; ".join(parts)
        if self.referenced_files:
            files = ", ".join(list(self.referenced_files)[-5:])
            context += f" [Files discussed: {files}]"

        return context


# In-memory conversation store (SQLite in production)
_conversations: dict[str, Conversation] = {}


def get_or_create_conversation(conversation_id: str | None = None) -> Conversation:
    if conversation_id and conversation_id in _conversations:
        return _conversations[conversation_id]

    conv = Conversation(conversation_id=conversation_id or str(uuid.uuid4()))
    _conversations[conv.conversation_id] = conv
    return conv
