"""
api/models.py
──────────────
Pydantic request/response models for the FastAPI endpoints.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# ── Request models ────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id:  str = Field(...,  description="Unique conversation/session ID")
    message:     str = Field(...,  description="User's question or message")
    new_session: bool = Field(False, description="Force-start a new session context")


# ── Response models ───────────────────────────────────────────────────────────

class CitationModel(BaseModel):
    index:        int
    id:           str
    title:        str
    source_type:  str           # 'research' | 'guideline'
    condition:    str
    url:          str
    pub_date:     Optional[str] = None
    pmid:         Optional[str] = None
    snippet:      str           # first 200 chars of chunk
    rerank_score: float = 0.0


class ChatResponse(BaseModel):
    session_id:           str
    answer:               str
    citations:            list[CitationModel]
    route:                str           # 'research' | 'guideline' | 'both'
    is_refused:           bool
    refuse_reason:        Optional[str] = None
    has_contradiction:    bool
    contradiction_details: Optional[str] = None
    rewrite_count:        int = 0
    docs_retrieved:       int = 0


class SessionSummary(BaseModel):
    session_id:    str
    message_count: int
    preview:       str   # first user message or last question


class HistoryMessage(BaseModel):
    role:      str   # 'human' | 'ai'
    content:   str
    citations: list[CitationModel] = []


class SessionHistoryResponse(BaseModel):
    session_id: str
    messages:   list[HistoryMessage]


class HealthResponse(BaseModel):
    status:     str
    qdrant:     str
    groq_keys:  int
    bm25_index: str


# ── Auth models ───────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=40)
    password: str = Field(..., min_length=6)


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    username:     str


class UserPublic(BaseModel):
    id:       int
    username: str
