"""
agent/nodes/intake.py
──────────────────────
Retrieval gating node: decides whether the current user message
actually requires the RAG pipeline or can be answered conversationally.

This prevents the full embedding + BM25 + reranking pipeline from
firing on greetings, thank-you messages, and simple clarifications.

Decision:
  RAG  → medical/factual question needing document retrieval
  CHAT → greeting, chit-chat, meta-question, or answerable from context

The node receives the full history_context so it can correctly classify
ambiguous follow-ups like "tell me more" (RAG if prior topic is medical)
or "what did you just say?" (CHAT, answerable from context alone).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from agent.llm import get_llm
from agent.state import AgentState


_INTAKE_PROMPT = """You are a medical chatbot router. Classify whether the user's message requires searching a medical document database (RAG) or can be answered conversationally without retrieval (CHAT).

Conversation context:
{history_context}

Current user message: {query}

Classification rules:
- RAG  → medical or health question needing facts, research, or guidelines
- RAG  → follow-up that needs more document evidence (e.g. "tell me more about that treatment")
- CHAT → greeting, thanks, general chit-chat (e.g. "hi", "thanks", "okay")
- CHAT → question answerable purely from the conversation context above
- CHAT → meta questions about the conversation (e.g. "what was my first question?")

Examples:
- "hi there" → CHAT
- "thanks!" → CHAT
- "what is type 2 diabetes?" → RAG
- "what are SGLT2 inhibitor side effects?" → RAG
- "what did you say about metformin?" → CHAT
- "tell me more" → RAG  (when prior topic was medical)
- "can you summarize what we discussed?" → CHAT

Respond with ONLY one word: RAG or CHAT
"""


def intake_node(state: AgentState) -> dict:
    """Decide whether the query needs RAG or can be answered conversationally."""
    query = state.get("query", "")
    history_context = state.get("history_context", "No prior conversation.")

    try:
        llm = get_llm(temperature=0.0, max_tokens=5)
        result = llm.invoke(
            _INTAKE_PROMPT.format(
                query=query,
                history_context=history_context or "No prior conversation.",
            )
        )
        decision = result.content.strip().upper()
        needs_rag = "CHAT" not in decision  # default to RAG if ambiguous
    except Exception as exc:
        print(f"[WARN] intake_node failed: {exc} — defaulting to RAG")
        needs_rag = True

    path = "RAG" if needs_rag else "CHAT"
    print(f"[intake] '{query[:60]}' → {path}")
    return {"needs_rag": needs_rag}
