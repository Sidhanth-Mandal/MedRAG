"""
agent/nodes/answer.py
──────────────────────
Answer generation node: produces the final answer with
inline citations, and optionally a conflict disclaimer.

Citations are formatted as [1], [2], etc. inline, with
the full citation list returned separately as structured dicts.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from agent.llm import get_llm
from agent.state import AgentState


_ANSWER_PROMPT = """
You are a medical information assistant. Answer the question using ONLY
the provided sources. Do not use any external knowledge.

{history_section}
Current question: {query}

Sources:
{sources_text}

Instructions:
1. Write a clear, accurate answer for a health-literate general audience.
2. Cite sources inline as [1], [2], etc. every time you use information from them.
3. Do not make claims not supported by the sources.
4. Keep the answer focused and under 300 words.
5. If the sources mention limitations or uncertainty, include that.
6. If the current question is a follow-up, use the conversation context above to understand what it refers to.
{conflict_instruction}

Answer:
"""

_CONFLICT_INSTRUCTION = """
6. IMPORTANT: The sources contain conflicting information: {conflict_details}
   Explicitly acknowledge this disagreement in your answer rather than
   choosing one view silently.
"""


def _format_sources(docs: list[dict[str, Any]]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        title = doc.get("title", "Unknown")[:80]
        source = doc.get("source_type", "")
        text   = doc.get("chunk_text", "")[:400]
        parts.append(f"[{i}] ({source.upper()}) {title}\n{text}")
    return "\n\n".join(parts)


def _build_citations(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    citations = []
    for i, doc in enumerate(docs, 1):
        pmid = doc.get("pmid")
        url  = doc.get("url", "")
        if pmid and not url:
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        citations.append({
            "index":       i,
            "id":          doc.get("doc_id") or doc.get("chunk_id", ""),
            "title":       doc.get("title", ""),
            "source_type": doc.get("source_type", ""),
            "condition":   doc.get("condition", ""),
            "url":         url,
            "pub_date":    doc.get("pub_date", ""),
            "pmid":        pmid,
            "snippet":     doc.get("chunk_text", "")[:200],
            "rerank_score": doc.get("rerank_score", 0.0),
        })
    return citations


def answer_node(state: AgentState) -> dict:
    """Generate the final answer with inline citations."""
    query            = state["query"]
    docs             = state.get("retrieved_docs", [])
    has_conflict     = state.get("has_contradiction", False)
    conflict_details = state.get("contradiction_details", "")
    history_context  = state.get("history_context", "")

    # Build the history section if we have prior context
    if history_context:
        history_section = (
            "Conversation context (prior turns):\n"
            f"{history_context}\n"
        )
    else:
        history_section = ""

    if not docs:
        return {
            "answer":    "I could not find relevant information to answer this question. "
                         "Please try rephrasing or consult a healthcare professional.",
            "citations": [],
            "messages":  [AIMessage(content="No relevant documents found.")],
        }

    sources_text = _format_sources(docs)
    conflict_instruction = (
        _CONFLICT_INSTRUCTION.format(conflict_details=conflict_details)
        if has_conflict
        else ""
    )

    prompt = _ANSWER_PROMPT.format(
        query=query,
        sources_text=sources_text,
        conflict_instruction=conflict_instruction,
        history_section=history_section,
    )

    llm    = get_llm()
    result = llm.invoke(prompt)
    answer = result.content.strip()

    citations = _build_citations(docs)

    # Add to message history
    new_messages = [
        HumanMessage(content=query),
        AIMessage(content=answer),
    ]

    return {
        "answer":    answer,
        "citations": citations,
        "messages":  new_messages,
    }

