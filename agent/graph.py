"""
agent/graph.py
───────────────
Assembles the LangGraph state machine with all nodes and
conditional edges. Compiles with a Supabase/PostgreSQL
checkpointer for persistent chat history across sessions.

Graph flow:
  summarize → intake → (CHAT?) → direct_answer → END
                     → (RAG?)  → safety → (unsafe?) → END
                                        → (safe?)   → router → query_rewrite
                                                             → (single?) → retrieve
                                                             → (multi?)  → multi_retrieve
                                                                    ↕ grade retry loop
                                          contradiction → answer → END
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.state import AgentState
from agent.nodes.summarizer    import summarizer_node
from agent.nodes.intake        import intake_node
from agent.nodes.safety        import safety_node
from agent.nodes.router        import router_node
from agent.nodes.query_rewriter import query_rewriter_node
from agent.nodes.retriever     import retriever_node
from agent.nodes.multi_retriever import multi_retriever_node
from agent.nodes.grader        import grader_node
from agent.nodes.contradiction import contradiction_node
from agent.nodes.answer        import answer_node
from agent.nodes.direct_answer import direct_answer_node
from config import cfg

load_dotenv()


# ── Conditional edge functions ────────────────────────────────────────────────

def after_intake(state: AgentState) -> str:
    """Route after intake: skip RAG for conversational turns."""
    return "rag" if state.get("needs_rag", True) else "conversational"


def after_safety(state: AgentState) -> str:
    """Route after safety check."""
    if not state.get("is_safe", True):
        return "refused"
    return "safe"


def after_query_rewrite(state: AgentState) -> str:
    """Route after query rewriting: single query → retrieve, multi → multi_retrieve."""
    queries = state.get("search_queries", [])
    return "multi" if len(queries) > 1 else "single"


def after_grade(state: AgentState) -> str:
    """Route after grader: retry retrieval or proceed."""
    relevant      = state.get("docs_relevant", True)
    rewrite_count = state.get("rewrite_count", 0)

    if not relevant and rewrite_count < cfg.agent.max_rewrite_retries:
        return "retry"
    return "proceed"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(checkpointer=None):
    """
    Build and compile the LangGraph agent.

    Args:
        checkpointer: A LangGraph checkpointer (e.g. PostgresSaver).
                      If None, graph runs without persistence.

    Returns:
        Compiled LangGraph app.
    """
    builder = StateGraph(AgentState)

    # Register all nodes
    builder.add_node("summarize",      summarizer_node)
    builder.add_node("intake",         intake_node)
    builder.add_node("direct_answer",  direct_answer_node)
    builder.add_node("safety",         safety_node)
    builder.add_node("router",         router_node)
    builder.add_node("query_rewrite",  query_rewriter_node)
    builder.add_node("retrieve",       retriever_node)
    builder.add_node("multi_retrieve", multi_retriever_node)
    builder.add_node("grade",          grader_node)
    builder.add_node("contradiction",  contradiction_node)
    builder.add_node("answer",         answer_node)

    # Entry point
    builder.set_entry_point("summarize")

    # summarize → intake (always)
    builder.add_edge("summarize", "intake")

    # intake → direct_answer OR safety (RAG path)
    builder.add_conditional_edges(
        "intake",
        after_intake,
        {
            "conversational": "direct_answer",
            "rag":            "safety",
        },
    )

    # Conversational fast-path → END
    builder.add_edge("direct_answer", END)

    # Safety → router OR END (refused)
    builder.add_conditional_edges(
        "safety",
        after_safety,
        {
            "safe":    "router",
            "refused": END,
        },
    )

    # Router → query_rewrite (always)
    builder.add_edge("router", "query_rewrite")

    # query_rewrite → single retrieve OR multi_retrieve
    builder.add_conditional_edges(
        "query_rewrite",
        after_query_rewrite,
        {
            "single": "retrieve",
            "multi":  "multi_retrieve",
        },
    )

    # Both retrieval paths → grade
    builder.add_edge("retrieve",       "grade")
    builder.add_edge("multi_retrieve", "grade")

    # Grade → retry retrieve OR proceed to contradiction
    builder.add_conditional_edges(
        "grade",
        after_grade,
        {
            "retry":   "retrieve",   # retry uses single retriever with rewritten search_query
            "proceed": "contradiction",
        },
    )

    # Contradiction → answer (always)
    builder.add_edge("contradiction", "answer")

    # Answer → END
    builder.add_edge("answer", END)

    # Compile
    if checkpointer:
        return builder.compile(checkpointer=checkpointer)
    return builder.compile()


# ── Checkpointer setup ────────────────────────────────────────────────────────

# Holds the open context manager so the DB connection isn't garbage-collected
_checkpointer_ctx = None


def get_postgres_checkpointer():
    """
    Create a PostgreSQL checkpointer using the Supabase connection string.
    Returns None if SUPABASE_DB_URL is not set (run without persistence).

    Note: PostgresSaver.from_conn_string() returns a context manager in newer
    versions of langgraph-checkpoint-postgres. We enter it here and keep a
    reference so the underlying connection stays alive.
    """
    global _checkpointer_ctx

    db_url = os.environ.get("SUPABASE_DB_URL", "")
    if not db_url:
        print("[WARN] SUPABASE_DB_URL not set — running without chat history persistence.")
        return None

    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        conn_string = db_url
        ctx = PostgresSaver.from_conn_string(conn_string)

        # Enter the context manager to get the real checkpointer object
        checkpointer = ctx.__enter__()
        _checkpointer_ctx = ctx  # keep alive for the process lifetime

        checkpointer.setup()  # creates langgraph checkpoint tables if needed
        print("[OK] PostgreSQL checkpointer initialized (Supabase)")
        return checkpointer
    except Exception as exc:
        print(f"[WARN] Could not initialize checkpointer: {exc}")
        print("[WARN] Running without persistence.")
        return None


# ── History context builder ───────────────────────────────────────────────────

def _build_history_context(session_id: str) -> tuple[str, str]:
    """
    Build the history_context string to inject into agent prompts.

    Returns (chat_summary, history_context) where:
      - chat_summary: the stored rolling summary (or "")
      - history_context: formatted string of summary + recent raw messages

    Strategy:
      - If a summary exists (history was long enough to trigger summarization):
          Use summary + last 4 raw messages
      - Otherwise:
          Use last 6 raw messages only
    """
    try:
        import api.db as db
        summary       = db.get_summary(session_id)
        recent_limit  = 4 if summary else 6
        recent_msgs   = db.get_recent_history(session_id, limit=recent_limit)
    except Exception as exc:
        print(f"[WARN] _build_history_context failed: {exc}")
        return "", ""

    parts: list[str] = []

    if summary:
        parts.append(f"[Summary of earlier conversation]\n{summary}")

    if recent_msgs:
        lines = []
        for msg in recent_msgs:
            role    = "User" if msg["role"] == "human" else "Assistant"
            content = msg["content"][:400]
            lines.append(f"{role}: {content}")
        parts.append("[Recent messages]\n" + "\n".join(lines))

    history_context = "\n\n".join(parts) if parts else ""
    return summary or "", history_context


# ── Convenience: invoke with session ─────────────────────────────────────────

def run_agent(
    query: str,
    session_id: str = "default",
    app=None,
) -> dict:
    """
    Run the agent for a given query and session.

    Args:
        query:      User's question
        session_id: Conversation thread ID (for checkpointing)
        app:        Pre-compiled graph (built lazily if None)

    Returns:
        Final AgentState dict with answer, citations, etc.
    """
    global _app
    if app is None:
        if _app is None:
            _app = _build_default_app()
        app = _app

    config = {"configurable": {"thread_id": session_id}}

    # Build history context from Postgres before the graph runs
    chat_summary, history_context = _build_history_context(session_id)

    # IMPORTANT: Only pass the fresh query fields here.
    # LangGraph merges this initial_state with any stored checkpoint for the
    # thread_id.  Non-message fields that must be fresh for every turn are
    # reset here; retrieved_docs / answer / citations are produced by the graph
    # nodes so they must also be reset to avoid a previous run's values
    # bleeding through.
    initial_state: dict[str, Any] = {
        # ── Per-turn inputs (always fresh) ──────────────────────────
        "query":                 query,
        "session_id":            session_id,
        # ── Context (built from DB before graph runs) ────────────────
        "chat_summary":          chat_summary,
        "history_context":       history_context,
        # ── Per-turn outputs (reset so previous run can't bleed in) ─
        "needs_rag":             True,
        "route":                 "both",
        "search_query":          query,   # fallback; query_rewriter overwrites this
        "search_queries":        [],
        "retrieved_docs":        [],
        "rewrite_count":         0,
        "docs_relevant":         False,
        "is_safe":               True,
        "is_refused":            False,
        "refuse_reason":         "",
        "has_contradiction":     False,
        "contradiction_details": "",
        "answer":                "",
        "citations":             [],
        # ── messages is intentionally OMITTED ───────────────────────
        # The add_messages reducer appends; let the checkpointer manage
        # the history.  Passing [] here would reset the thread's memory.
    }

    final_state = app.invoke(initial_state, config=config)
    return final_state


# ── Module-level lazy app ─────────────────────────────────────────────────────

_app = None


def _build_default_app():
    checkpointer = get_postgres_checkpointer()
    return build_graph(checkpointer=checkpointer)


def get_app():
    global _app
    if _app is None:
        _app = _build_default_app()
    return _app
