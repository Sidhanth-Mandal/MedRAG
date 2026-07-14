"""
agent/state.py
───────────────
Defines the LangGraph AgentState TypedDict — the central
data structure that flows through every node in the graph.
"""

from __future__ import annotations

from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # ── Input ───────────────────────────────────────────
    query:           str          # Current user question
    session_id:      str          # Thread/session identifier

    # ── Chat history (managed by LangGraph add_messages) ─
    messages:        Annotated[list, add_messages]

    # ── Routing ─────────────────────────────────────────
    route:           str          # 'research' | 'guideline' | 'both'

    # ── Retrieval ────────────────────────────────────────
    retrieved_docs:  list[dict[str, Any]]   # Top chunks after reranking
    search_query:    str          # Possibly rewritten query used for retrieval

    # ── Grading / retry ──────────────────────────────────
    rewrite_count:   int          # 0, 1, or 2 — how many rewrites attempted
    docs_relevant:   bool         # Did the grader accept the docs?

    # ── Safety ───────────────────────────────────────────
    is_safe:         bool         # False → refuse, don't answer
    refuse_reason:   str          # Populated when is_safe=False

    # ── Contradiction ────────────────────────────────────
    has_contradiction:     bool
    contradiction_details: str    # Human-readable description of conflict

    # ── Output ──────────────────────────────────────────
    answer:          str          # Final answer text
    citations:       list[dict[str, Any]]   # Citation objects
    is_refused:      bool         # True when safety node refuses
