"""
retrieval/dense_retriever.py
──────────────────────────────
Thin wrapper for dense vector search via Qdrant.
Embeds query text with Cloudflare Workers AI and searches Qdrant.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import cfg
from pipeline.embedder import get_embedder
from pipeline.qdrant_uploader import get_qdrant_client, search as qdrant_search


def search(
    query: str,
    top_k: int = None,
    source_type: str | None = None,
    condition: str | None = None,
) -> list[dict[str, Any]]:
    """
    Dense vector search.

    Args:
        query:       Natural language query string
        top_k:       Number of results to return
        source_type: Filter to 'research' or 'guideline' (None = both)
        condition:   Filter to a specific condition slug

    Returns:
        List of chunk dicts with added 'dense_score' field.
    """
    top_k = top_k or cfg.qdrant.dense_top_k

    embedder = get_embedder()
    query_vector = embedder.embed_one(query)

    client = get_qdrant_client()
    results = qdrant_search(
        client,
        query_vector=query_vector,
        top_k=top_k,
        source_type=source_type,
        condition=condition,
    )

    # Rename 'score' → 'dense_score' for clarity in fusion
    for r in results:
        r["dense_score"] = r.pop("score", 0.0)

    return results
