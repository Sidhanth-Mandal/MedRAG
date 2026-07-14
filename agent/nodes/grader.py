"""
agent/nodes/grader.py
──────────────────────
Self-RAG style grader node:
  - Checks whether retrieved chunks actually answer the question
  - If not relevant: rewrites the search query (max 2 retries)
  - If still not good after max retries: proceeds anyway
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from agent.llm import get_llm
from agent.state import AgentState
from config import cfg


_GRADE_PROMPT = """
You are evaluating whether retrieved medical documents answer a user's question.

Question: {query}

Retrieved excerpts:
{docs_text}

Do these excerpts contain information that is directly relevant to answering
the question? Consider a 'yes' if at least 2 excerpts are relevant.

Answer with ONLY: yes | no
"""

_REWRITE_PROMPT = """
You are improving a medical search query to find better results.

Original question: {query}
Previous search query: {search_query}
The previous search returned irrelevant results.

Rewrite the search query to be more specific and likely to find relevant
medical information. Return ONLY the new search query, nothing else.
"""


def grader_node(state: AgentState) -> dict:
    """Grade retrieved docs for relevance; rewrite query if needed."""
    query         = state["query"]
    search_query  = state.get("search_query") or query
    docs          = state.get("retrieved_docs", [])
    rewrite_count = state.get("rewrite_count", 0)

    if not docs:
        # Nothing to grade — trigger rewrite if budget allows
        if rewrite_count < cfg.agent.max_rewrite_retries:
            return {
                "docs_relevant":  False,
                "rewrite_count":  rewrite_count + 1,
                "search_query":   _rewrite_query(query, search_query),
            }
        return {"docs_relevant": False}

    # Format top-5 chunks for the grader
    docs_text = "\n\n".join(
        f"[{i+1}] {d['chunk_text'][:300]}"
        for i, d in enumerate(docs[:5])
    )

    try:
        llm    = get_llm(temperature=0.0, max_tokens=5)
        result = llm.invoke(
            _GRADE_PROMPT.format(query=query, docs_text=docs_text)
        )
        verdict = result.content.strip().lower()
        relevant = "yes" in verdict
    except Exception:
        relevant = True  # on error, proceed with what we have

    if relevant or rewrite_count >= cfg.agent.max_rewrite_retries:
        return {
            "docs_relevant": relevant,
            "rewrite_count": rewrite_count,
        }

    # Not relevant — rewrite and retry
    new_query = _rewrite_query(query, search_query)
    return {
        "docs_relevant":  False,
        "rewrite_count":  rewrite_count + 1,
        "search_query":   new_query,
    }


def _rewrite_query(original: str, previous: str) -> str:
    """Ask the LLM to rewrite the search query."""
    try:
        llm    = get_llm(temperature=0.3, max_tokens=60)
        result = llm.invoke(
            _REWRITE_PROMPT.format(
                query=original,
                search_query=previous,
            )
        )
        return result.content.strip()
    except Exception:
        return original  # fallback: use original
