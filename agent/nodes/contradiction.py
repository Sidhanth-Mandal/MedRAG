"""
agent/nodes/contradiction.py
─────────────────────────────
Contradiction check node: examines retrieved chunks to see
if research sources disagree with each other on key claims.
Flags it explicitly in the state so the answer node can
include a disclaimer rather than silently picking one view.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from agent.llm import get_llm
from agent.state import AgentState


_CONTRADICTION_PROMPT = """
You are checking whether medical sources contradict each other.

User question: {query}

Retrieved passages:
{docs_text}

Do any of these passages make contradictory or conflicting claims about
the same medical topic relevant to the question? Look for:
- Conflicting treatment recommendations
- Opposite findings on drug efficacy or safety
- Disagreement on targets or thresholds (e.g. blood pressure goals)

If YES, briefly describe the conflict in 1-2 sentences.
If NO, say only: NO_CONFLICT

Response format:
YES: <brief description of the conflict>
OR
NO_CONFLICT
"""


def contradiction_node(state: AgentState) -> dict:
    """Detect contradictions among retrieved chunks."""
    docs  = state.get("retrieved_docs", [])
    query = state["query"]

    if len(docs) < 2:
        return {"has_contradiction": False, "contradiction_details": ""}

    docs_text = "\n\n".join(
        f"[Source {i+1} | {d['source_type']} | {d.get('title','')[:50]}]\n"
        f"{d['chunk_text'][:350]}"
        for i, d in enumerate(docs[:6])
    )

    try:
        llm    = get_llm(temperature=0.1, max_tokens=120)
        result = llm.invoke(
            _CONTRADICTION_PROMPT.format(query=query, docs_text=docs_text)
        )
        response = result.content.strip()

        if response.upper().startswith("YES"):
            details = response[4:].strip() if len(response) > 4 else "Sources conflict."
            return {
                "has_contradiction":     True,
                "contradiction_details": details,
            }
    except Exception:
        pass

    return {"has_contradiction": False, "contradiction_details": ""}
