"""
api/main.py
────────────
FastAPI backend for the Agentic RAG Medical Q&A chatbot.

Endpoints:
  POST   /api/auth/register                  — Create a new account
  POST   /api/auth/login                     — Login, receive JWT
  GET    /api/auth/verify                    — Verify token, return current user
  POST   /api/chat                           — Send a message (auth required)
  GET    /api/sessions                       — List user's sessions (auth required)
  GET    /api/sessions/{session_id}/history  — Full history for a session (auth required)
  DELETE /api/sessions/{session_id}          — Delete a session (auth required)
  GET    /api/health                         — Health check
  GET    /api/stats                          — Database stats

Usage:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import FileResponse
from jose import JWTError, jwt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from api.models import (
    ChatRequest, ChatResponse, CitationModel,
    SessionSummary, SessionHistoryResponse, HistoryMessage,
    HealthResponse,
    RegisterRequest, LoginRequest, TokenResponse, UserPublic,
)
from api import db as chat_db
from agent.graph import get_app, run_agent
from agent.llm import get_key_count
from config import cfg

# ── JWT config ────────────────────────────────────────────────────────────────

JWT_SECRET      = os.environ.get("JWT_SECRET", "change_me_please_use_a_real_secret")
JWT_ALGORITHM   = os.environ.get("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_DAYS = int(os.environ.get("JWT_EXPIRE_DAYS", "30"))

_bearer = HTTPBearer(auto_error=False)


def _create_token(user_id: int, username: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=JWT_EXPIRE_DAYS)
    return jwt.encode(
        {"sub": str(user_id), "username": username, "exp": expire},
        JWT_SECRET,
        algorithm=JWT_ALGORITHM,
    )


def _decode_token(token: str) -> dict:
    """Raise HTTPException 401 if invalid."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        if "sub" not in payload:
            raise ValueError("missing sub")
        return payload
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict:
    """FastAPI dependency — returns the authenticated user dict."""
    if not creds:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload  = _decode_token(creds.credentials)
    user_id  = int(payload["sub"])
    user     = chat_db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Agentic RAG - Medical Q&A",
    description="AI-powered medical Q&A over PubMed abstracts and MedlinePlus guidelines",
    version="1.0.0",
)

_raw_origins = os.environ.get(
    "ALLOWED_ORIGINS",
    "https://sidhanth-mandal.github.io,http://localhost:8000,http://127.0.0.1:8000"
)
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

import logging
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)
_log.info("CORS allow_origins: %s", ALLOWED_ORIGINS)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    max_age=3600,
)

# Not Necessaru now as frontent is now on github Pages
# # Serve frontend static files
# frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
# if frontend_dir.exists():
#     app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


# ── LangGraph app (lazy init) ─────────────────────────────────────────────────
_agent_app = None


def _get_agent():
    global _agent_app
    if _agent_app is None:
        _agent_app = get_app()
    return _agent_app


# ── DB schema bootstrap ───────────────────────────────────────────────────────
@app.on_event("startup")
async def _startup():
    """Ensure DB tables exist on boot."""
    chat_db.ensure_schema()


# # ── Frontend ──────────────────────────────────────────────────────────────────

# @app.get("/", include_in_schema=False)
# async def serve_frontend():
#     index = frontend_dir / "index.html"
#     if index.exists():
#         return FileResponse(str(index))
#     return {"message": "Agentic RAG API is running. Frontend not found at /frontend."}


# ── Auth routes ───────────────────────────────────────────────────────────────

@app.post("/api/auth/register", response_model=TokenResponse, tags=["Auth"])
async def register(req: RegisterRequest):
    """Register a new user and return a JWT."""
    user = chat_db.create_user(req.username, req.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already taken",
        )
    token = _create_token(user["id"], user["username"])
    return TokenResponse(access_token=token, username=user["username"])


@app.post("/api/auth/login", response_model=TokenResponse, tags=["Auth"])
async def login(req: LoginRequest):
    """Login with username and password, receive a JWT."""
    user = chat_db.get_user_by_username(req.username)
    if not user or not chat_db.verify_password(req.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )
    token = _create_token(user["id"], user["username"])
    return TokenResponse(access_token=token, username=user["username"])


@app.get("/api/auth/verify", response_model=UserPublic, tags=["Auth"])
async def verify_token(current_user: dict = Depends(get_current_user)):
    """Verify the bearer token and return the current user."""
    return UserPublic(id=current_user["id"], username=current_user["username"])


# ── Health / Stats ────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check the health of all system components."""
    qdrant_status = "unknown"
    try:
        from pipeline.qdrant_uploader import get_qdrant_client
        client     = get_qdrant_client()
        col_names  = [c.name for c in client.get_collections().collections]
        col_name   = os.environ.get("QDRANT_COLLECTION_NAME") or cfg.qdrant.collection_name
        if col_name in col_names:
            info   = client.get_collection(col_name)
            points = getattr(info, "points_count", "?")
            qdrant_status = f"ok ({points} points in '{col_name}')"
        else:
            qdrant_status = "ok (collection not found — run pipeline first)"
    except Exception as exc:
        qdrant_status = f"error: {str(exc)[:60]}"

    bm25_status = "index not built"
    bm25_path   = Path("data/bm25_index.pkl")
    if bm25_path.exists():
        bm25_status = f"ok ({bm25_path.stat().st_size // 1024} KB)"

    return HealthResponse(
        status="ok",
        qdrant=qdrant_status,
        groq_keys=get_key_count(),
        bm25_index=bm25_status,
    )


@app.get("/api/stats", tags=["System"])
async def get_db_stats():
    """Return document statistics from the Qdrant vector store."""
    try:
        from pipeline.qdrant_uploader import get_qdrant_client
        from qdrant_client.http import models as qmodels

        client   = get_qdrant_client()
        col_name = os.environ.get("QDRANT_COLLECTION_NAME") or cfg.qdrant.collection_name

        info         = client.get_collection(col_name)
        total_points = getattr(info, "points_count", 0) or 0

        def count_by(field: str, value: str) -> int:
            result = client.count(
                collection_name=col_name,
                count_filter=qmodels.Filter(
                    must=[qmodels.FieldCondition(key=field, match=qmodels.MatchValue(value=value))]
                ),
                exact=True,
            )
            return result.count

        return {
            "total_documents": total_points,
            "by_source": {
                "research":  count_by("source_type", "research"),
                "guideline": count_by("source_type", "guideline"),
            },
            "by_condition": {
                "diabetes":     count_by("condition", "type2_diabetes"),
                "hypertension": count_by("condition", "hypertension"),
                "asthma":       count_by("condition", "asthma"),
            },
            "collection": col_name,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Stats error: {str(exc)}")


# ── Chat ──────────────────────────────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    """Send a message and get an AI answer with inline citations."""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    agent = _get_agent()

    try:
        result = run_agent(
            query=request.message.strip(),
            session_id=request.session_id,
            app=agent,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(exc)}")

    citations = [
        CitationModel(
            index=c["index"],
            id=c["id"],
            title=c["title"],
            source_type=c["source_type"],
            condition=c.get("condition", ""),
            url=c["url"],
            pub_date=c.get("pub_date"),
            pmid=c.get("pmid"),
            snippet=c.get("snippet", "")[:200],
            rerank_score=c.get("rerank_score", 0.0),
        )
        for c in result.get("citations", [])
    ]

    session_id = request.session_id
    user_id    = current_user["id"]

    chat_db.save_message(
        session_id=session_id,
        user_id=user_id,
        role="human",
        content=request.message,
    )
    chat_db.save_message(
        session_id=session_id,
        user_id=user_id,
        role="ai",
        content=result.get("answer", ""),
        citations=[c.model_dump() for c in citations],
        route=result.get("route", "both"),
        is_refused=result.get("is_refused", False),
        has_contradiction=result.get("has_contradiction", False),
        contradiction_details=result.get("contradiction_details"),
    )

    return ChatResponse(
        session_id=session_id,
        answer=result.get("answer", ""),
        citations=citations,
        route=result.get("route", "both"),
        is_refused=result.get("is_refused", False),
        refuse_reason=result.get("refuse_reason"),
        has_contradiction=result.get("has_contradiction", False),
        contradiction_details=result.get("contradiction_details"),
        rewrite_count=result.get("rewrite_count", 0),
        docs_retrieved=len(result.get("retrieved_docs", [])),
    )


# ── Sessions ──────────────────────────────────────────────────────────────────

@app.get("/api/sessions", response_model=list[SessionSummary], tags=["Sessions"])
async def list_sessions(current_user: dict = Depends(get_current_user)):
    """List sessions for the authenticated user."""
    rows = chat_db.list_sessions(user_id=current_user["id"])
    return [
        SessionSummary(
            session_id=r["session_id"],
            message_count=r["message_count"],
            preview=r["preview"] or "",
        )
        for r in rows
    ]


@app.get(
    "/api/sessions/{session_id}/history",
    response_model=SessionHistoryResponse,
    tags=["Sessions"],
)
async def get_session_history(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Get the full chat history for a session."""
    rows = chat_db.get_history(session_id)
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found",
        )
    messages = [
        HistoryMessage(
            role=msg["role"],
            content=msg["content"],
            citations=[CitationModel(**c) for c in (msg.get("citations") or [])],
        )
        for msg in rows
    ]
    return SessionHistoryResponse(session_id=session_id, messages=messages)


@app.delete("/api/sessions/{session_id}", tags=["Sessions"])
async def delete_session(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Delete a session and its chat history."""
    chat_db.delete_session(session_id)
    return {"message": f"Session '{session_id}' deleted"}


# ── Dev server entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(Path(__file__).resolve().parent.parent)],
    )
