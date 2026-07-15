"""
agent/nodes/summarizer.py
──────────────────────────
Summarization node: fires at the start of every turn.

If the raw message count for this session has reached or exceeded
cfg.agent.chat_history_window (default 10), it asks the LLM to produce
a rolling ~150-word summary of the conversation so far and saves it to
the session_summaries table in Postgres.

This keeps the history_context prompt section concise regardless of how
long the conversation has been running.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from agent.llm import get_llm
from agent.state import AgentState
from config import cfg

# Lazy import to avoid circular dependency at module load time
def _get_db():
    import api.db as db
    return db


_SUMMARIZE_PROMPT = """
You are summarizing a medical Q&A conversation for a memory system.

Conversation history:
{history_text}

Write a concise summary (max 150 words) that captures:
- The medical topics discussed (conditions, drugs, treatments)
- Key facts or answers provided
- Any follow-up threads or unresolved questions

Summary:
"""


def summarizer_node(state: AgentState) -> dict:
    """
    Check message count; if >= threshold, generate and store a rolling summary.
    Always returns the current chat_summary (from state or freshly generated).
    """
    session_id = state.get("session_id", "")
    if not session_id:
        return {}

    db = _get_db()

    # Count raw messages in DB for this session
    msg_count = db.count_messages(session_id)

    if msg_count < cfg.agent.chat_history_window:
        # Not enough history yet — nothing to summarize
        return {"chat_summary": state.get("chat_summary", "")}

    # Fetch full history to summarize
    try:
        full_history = db.get_history(session_id)
    except Exception:
        return {"chat_summary": state.get("chat_summary", "")}

    if not full_history:
        return {"chat_summary": state.get("chat_summary", "")}

    # Format for the LLM
    lines = []
    for msg in full_history:
        role = "User" if msg["role"] == "human" else "Assistant"
        content = msg["content"][:300]  # truncate very long messages
        lines.append(f"{role}: {content}")
    history_text = "\n".join(lines)

    try:
        llm = get_llm(temperature=0.2, max_tokens=250)
        result = llm.invoke(_SUMMARIZE_PROMPT.format(history_text=history_text))
        summary = result.content.strip()
        db.save_summary(session_id, summary)
        print(f"[summarizer] Summary saved for session '{session_id}' ({msg_count} msgs)")
        return {"chat_summary": summary}
    except Exception as exc:
        print(f"[WARN] summarizer_node failed: {exc}")
        return {"chat_summary": state.get("chat_summary", "")}
