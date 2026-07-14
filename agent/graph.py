"""
agent/graph.py
───────────────
Assembles the LangGraph state machine with all nodes and
conditional edges. Compiles with a Supabase/PostgreSQL
checkpointer for persistent chat history across sessions.

Graph flow:
  safety → (safe?) → router → retrieve → grade
  grade → (relevant?) → contradiction → answer → END
  grade → (retry?)    → retrieve  [max 2 retries]
  safety → (unsafe?) → END  [with refuse answer already set]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.state import AgentState
from agent.nodes.safety      import safety_node
from agent.nodes.router      import router_node
from agent.nodes.retriever   import retriever_node
from agent.nodes.grader      import grader_node
from agent.nodes.contradiction import contradiction_node
from agent.nodes.answer      import answer_node
from config import cfg

load_dotenv()


# ── Conditional edge functions ────────────────────────────────────────────────

def after_safety(state: AgentState) -> str:
    """Route after safety check."""
    if not state.get("is_safe", True):
        return "refused"
    return "safe"


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

    # Register nodes
    builder.add_node("safety",       safety_node)
    builder.add_node("router",       router_node)
    builder.add_node("retrieve",     retriever_node)
    builder.add_node("grade",        grader_node)
    builder.add_node("contradiction", contradiction_node)
    builder.add_node("answer",       answer_node)

    # Entry point
    builder.set_entry_point("safety")

    # Safety → router OR END
    builder.add_conditional_edges(
        "safety",
        after_safety,
        {
            "safe":    "router",
            "refused": END,
        },
    )

    # Router → retrieve (always)
    builder.add_edge("router", "retrieve")

    # Retrieve → grade (always)
    builder.add_edge("retrieve", "grade")

    # Grade → retry retrieve OR proceed to contradiction
    builder.add_conditional_edges(
        "grade",
        after_grade,
        {
            "retry":   "retrieve",
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

    # IMPORTANT: Only pass the fresh query fields here.
    # LangGraph merges this initial_state with any stored checkpoint for the
    # thread_id.  Passing stale-reset values (empty lists, False booleans, etc.)
    # would overwrite the checkpointer's memory of the current turn AND, without
    # a checkpointer, would leak state between different sessions sharing the
    # same in-process graph instance.
    # Non-message fields that must be fresh for every turn are reset here;
    # retrieved_docs / answer / citations are produced by the graph nodes so
    # they must also be reset to avoid a previous run's values bleeding through.
    initial_state = {
        # ── Per-turn inputs (always fresh) ──────────────────────────
        "query":                 query,
        "session_id":            session_id,
        "search_query":          query,   # may be rewritten by grader
        # ── Per-turn outputs (reset so previous run can't bleed in) ─
        "route":                 "both",
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
