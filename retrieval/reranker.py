"""
retrieval/reranker.py
──────────────────────
Cohere Rerank step: takes the hybrid-fused top-k chunks
and reranks them using a cross-encoder for better precision.

Model: rerank-english-v3.0 (configurable via config.py)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import cohere
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import cfg

load_dotenv()

_COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "")


# ── Client singleton ──────────────────────────────────────────────────────────

_client: cohere.Client | None = None


def get_cohere_client() -> cohere.Client:
    global _client
    if _client is None:
        if not _COHERE_API_KEY:
            raise ValueError("COHERE_API_KEY not set in .env")
        _client = cohere.Client(api_key=_COHERE_API_KEY)
    return _client


# ── Rerank ────────────────────────────────────────────────────────────────────

def rerank(
    query: str,
    chunks: list[dict[str, Any]],
    top_k: int = None,
) -> list[dict[str, Any]]:
    """
    Rerank a list of chunks using Cohere Rerank.

    Args:
        query:  The original user query
        chunks: List of chunk dicts (each must have 'chunk_text')
        top_k:  How many to return after reranking

    Returns:
        Reranked list of chunk dicts with added 'rerank_score' field.
    """
    top_k = top_k or cfg.retrieval.final_top_k

    if not chunks:
        return []

    # Cohere expects a list of strings as documents
    documents = [c["chunk_text"] for c in chunks]

    client = get_cohere_client()
    response = client.rerank(
        model=cfg.retrieval.cohere_rerank_model,
        query=query,
        documents=documents,
        top_n=min(top_k, len(documents)),
        return_documents=False,   # we have chunks already
    )

    reranked: list[dict[str, Any]] = []
    for result in response.results:
        chunk = dict(chunks[result.index])
        chunk["rerank_score"] = result.relevance_score
        reranked.append(chunk)

    return reranked
