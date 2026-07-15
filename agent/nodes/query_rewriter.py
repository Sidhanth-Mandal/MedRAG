"""
agent/nodes/query_rewriter.py
──────────────────────────────
Context-aware query rewriter with multi-query decomposition.

Runs after the router node, before retrieval. Does two jobs in one
LLM call:

Job 1 — Context resolution
  Reads the user query + history_context and resolves:
  - Pronouns ("it", "they", "the drug")  → the actual medical entity
  - Vague references ("the second one")  → the specific item mentioned
  - Implicit context ("side effects?")   → "metformin side effects diabetes"

Job 2 — Multi-query decomposition
  If the question spans multiple independent topics (comparison, multi-
  condition, multi-drug), the LLM emits 2–3 sub-queries so each angle
  gets good retrieval coverage independently.

Output format parsed:
  QUERIES:
  1. <sub-query 1>
  2. <sub-query 2>   ← only if needed
  3. <sub-query 3>   ← only if needed

Returns {"search_queries": [...], "search_query": <first_query>}
The search_query field is kept for backward-compat with grader rewriting.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from agent.llm import get_llm
from agent.state import AgentState


_REWRITER_PROMPT = """You are a medical search query optimizer for a RAG system.

Conversation context (prior turns):
{history_context}

Current user question: {query}

Your tasks:
1. Resolve any pronouns or vague references using the conversation context.
2. If the question is SIMPLE or asks about a SINGLE topic → produce exactly 1 search query.
3. If the question requires COMPARISON, MULTIPLE CONDITIONS, or MULTIPLE DRUGS → produce 2-3 focused sub-queries, one per topic/drug/condition.

Rules:
- Each query should be a crisp medical search phrase (no question marks, no filler words)
- Max 3 sub-queries
- Do NOT split a single-topic question unnecessarily

Output format (follow exactly):
QUERIES:
1. <query one>
2. <query two>
3. <query three>

Examples:
Question: "side effects?"  (prior context: discussing metformin)
QUERIES:
1. metformin side effects type 2 diabetes

Question: "compare metformin and SGLT2 inhibitors for diabetes"
QUERIES:
1. metformin efficacy type 2 diabetes clinical trials
2. SGLT2 inhibitor efficacy type 2 diabetes clinical trials

Question: "what treatments exist for both diabetes and hypertension?"
QUERIES:
1. type 2 diabetes treatment management
2. hypertension treatment management

Question: "how does exercise affect blood sugar and blood pressure?"
QUERIES:
1. exercise effect on blood glucose type 2 diabetes
2. exercise effect on blood pressure hypertension

Now produce the queries for the user question above:
"""


def _parse_queries(raw: str, fallback: str) -> list[str]:
    """Parse the QUERIES: block from LLM output."""
    queries = []
    # Find lines starting with a number followed by a dot or closing paren
    for line in raw.splitlines():
        m = re.match(r"^\s*\d+[.)]\s*(.+)", line)
        if m:
            q = m.group(1).strip()
            if q:
                queries.append(q)
    if not queries:
        # Fallback: just use the cleaned raw output as a single query
        cleaned = raw.replace("QUERIES:", "").strip()
        queries = [cleaned or fallback]
    return queries[:3]  # hard cap at 3


def query_rewriter_node(state: AgentState) -> dict:
    """Rewrite query with context resolution and optional multi-query decomposition."""
    query = state.get("query", "")
    history_context = state.get("history_context", "")

    try:
        llm = get_llm(temperature=0.1, max_tokens=150)
        result = llm.invoke(
            _REWRITER_PROMPT.format(
                query=query,
                history_context=history_context or "No prior conversation.",
            )
        )
        raw = result.content.strip()
        queries = _parse_queries(raw, fallback=query)
    except Exception as exc:
        print(f"[WARN] query_rewriter_node failed: {exc} — using original query")
        queries = [query]

    print(f"[query_rewriter] {len(queries)} query/queries: {queries}")

    return {
        "search_queries": queries,
        "search_query":   queries[0],   # primary query; grader uses this for retry rewriting
    }
