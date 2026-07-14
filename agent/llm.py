"""
agent/llm.py
─────────────
Groq LLM client with round-robin API key rotation.

Four keys are loaded from env (GROQ_API_KEY_1 through _4).
Each call to get_llm() returns the next LangChain ChatGroq
instance in rotation, spreading rate-limit pressure across keys.
"""

from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import cfg

load_dotenv()


def _load_keys() -> list[str]:
    keys = []
    for i in range(1, 5):
        k = os.environ.get(f"GROQ_API_KEY_{i}", "").strip()
        if k:
            keys.append(k)
    if not keys:
        raise ValueError(
            "No Groq API keys found. Set GROQ_API_KEY_1 .. GROQ_API_KEY_4 in .env"
        )
    return keys


_keys: list[str] = []
_key_cycle: itertools.cycle | None = None


def _get_next_key() -> str:
    global _keys, _key_cycle
    if not _keys:
        _keys = _load_keys()
        _key_cycle = itertools.cycle(_keys)
    return next(_key_cycle)


def get_llm(
    model: str = None,
    temperature: float = None,
    max_tokens: int = None,
) -> ChatGroq:
    """
    Return a ChatGroq instance using the next key in rotation.
    Parameters default to values in config.py.
    """
    return ChatGroq(
        api_key=_get_next_key(),
        model=model       or cfg.llm.model,
        temperature=temperature if temperature is not None else cfg.llm.temperature,
        max_tokens=max_tokens   or cfg.llm.max_tokens,
    )


def get_key_count() -> int:
    """Return how many keys are configured (for debugging)."""
    if not _keys:
        try:
            _load_keys()
        except ValueError:
            return 0
    return len(_keys)
