"""
agent/nodes/router.py
──────────────────────
Query router: classifies each question into one of three
retrieval strategies:
  'research'  → use PubMed abstracts only
  'guideline' → use MedlinePlus guidelines only
  'both'      → search both corpora
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from agent.llm import get_llm
from agent.state import AgentState

_ROUTER_PROMPT = """
You are a medical information router for a RAG system.

You have two data sources:
1. 'research'  — PubMed scientific abstracts (clinical trials, studies, mechanisms)
2. 'guideline' — MedlinePlus patient guidelines (what to do, how to manage, lifestyle)

User question: {query}

Decide which source(s) to search.
Respond with EXACTLY ONE of: research | guideline | both

Examples:
- "What does the research say about metformin and weight loss?" → research
- "What lifestyle changes help with high blood pressure?" → guideline
- "Is exercise effective for diabetes? What do guidelines recommend?" → both
- "What are the latest clinical trial results on SGLT2 inhibitors?" → research
- "How do I use an asthma inhaler correctly?" → guideline
- "What is asthma?" → both

Respond with only the category word, nothing else.
"""


def router_node(state: AgentState) -> dict:
    """Classify the query into research / guideline / both."""
    query = state["query"]

    try:
        llm    = get_llm(temperature=0.0, max_tokens=10)
        result = llm.invoke(_ROUTER_PROMPT.format(query=query))
        route  = result.content.strip().lower()

        # Sanitize output
        if route not in ("research", "guideline", "both"):
            route = "both"  # safe default
    except Exception:
        route = "both"

    # Note: search_query is set by query_rewriter_node (runs after this node).
    return {"route": route}

