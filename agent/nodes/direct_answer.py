"""
agent/nodes/direct_answer.py
─────────────────────────────
Direct answer node for the conversational (non-RAG) path.

When the intake node classifies a turn as CHAT, the full retrieval
pipeline is skipped. This node answers using only the conversation
history context — no documents, no citations.

Handles:
  - Greetings and chit-chat
  - Thank-you / acknowledgement messages
  - Meta questions ("what was my first question?")
  - Conversational clarifications answerable from history
"""

from __future__ import annotations

import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from agent.llm import get_llm
from agent.state import AgentState


_DIRECT_ANSWER_PROMPT = """You are a helpful medical information assistant.

Conversation history:
{history_context}

User message: {query}

Respond naturally and helpfully. If this is a greeting, respond warmly.
If the user is asking about something from the conversation history, answer using that context.
Keep your response concise and friendly.
Do NOT make up medical information — if a medical question needs document retrieval, say you'll look it up.

Response:
"""


def direct_answer_node(state: AgentState) -> dict:
    """Generate a conversational answer without RAG retrieval."""
    query          = state.get("query", "")
    history_context = state.get("history_context", "")

    try:
        llm = get_llm(temperature=0.4, max_tokens=300)
        result = llm.invoke(
            _DIRECT_ANSWER_PROMPT.format(
                query=query,
                history_context=history_context or "No prior conversation.",
            )
        )
        answer = result.content.strip()
    except Exception as exc:
        print(f"[WARN] direct_answer_node failed: {exc}")
        answer = "I'm sorry, I couldn't process that right now. Could you rephrase?"

    new_messages = [
        HumanMessage(content=query),
        AIMessage(content=answer),
    ]

    return {
        "answer":    answer,
        "citations": [],
        "is_refused": False,
        "messages":  new_messages,
    }
