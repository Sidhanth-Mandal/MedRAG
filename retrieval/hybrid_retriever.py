"""
retrieval/hybrid_retriever.py
──────────────────────────────
Hybrid retrieval: combines BM25 sparse + dense vector search
using Reciprocal Rank Fusion (RRF).

RRF formula: score(d) = sum_r( 1 / (k + rank_r(d)) )
where k=60 (config) and rank_r is the rank in each retriever's list.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import cfg
from retrieval import bm25_retriever
from retrieval import dense_retriever


def _rrf_score(rank: int, k: int = None) -> float:
    """Reciprocal Rank Fusion score for a single rank position."""
    k = k or cfg.retrieval.rrf_k
    return 1.0 / (k + rank + 1)   # +1 because ranks are 0-indexed


def search(
    query: str,
    source_type: str | None = None,
    condition: str | None = None,
    top_k: int = None,
) -> list[dict[str, Any]]:
    """
    Hybrid search: BM25 + dense → RRF fusion.

    Args:
        query:       Natural language query
        source_type: 'research' | 'guideline' | None (both)
        condition:   condition slug filter or None
        top_k:       Number of fused results to return

    Returns:
        List of chunk dicts sorted by RRF score (desc), with fields:
        - 'rrf_score': fused score
        - 'bm25_rank':   rank in BM25 results (-1 if absent)
        - 'dense_rank':  rank in dense results (-1 if absent)
    """
    top_k = top_k or cfg.retrieval.rerank_top_k
    rrf_k = cfg.retrieval.rrf_k

    # ── Run both retrievers ──────────────────────────────────
    bm25_model, chunks = bm25_retriever.get_bm25_index()

    bm25_results  = bm25_retriever.search(
        query, bm25_model, chunks,
        top_k=cfg.bm25.top_k,
        source_type=source_type,
        condition=condition,
    )
    dense_results = dense_retriever.search(
        query,
        top_k=cfg.qdrant.dense_top_k,
        source_type=source_type,
        condition=condition,
    )

    # ── Build lookup: chunk_id → (chunk_dict, scores) ────────
    fused: dict[str, dict[str, Any]] = {}

    for rank, chunk in enumerate(bm25_results):
        cid = chunk["chunk_id"]
        if cid not in fused:
            fused[cid] = dict(chunk)
            fused[cid]["rrf_score"]  = 0.0
            fused[cid]["bm25_rank"]  = -1
            fused[cid]["dense_rank"] = -1
        fused[cid]["rrf_score"] += _rrf_score(rank, rrf_k)
        fused[cid]["bm25_rank"]  = rank

    for rank, chunk in enumerate(dense_results):
        cid = chunk["chunk_id"]
        if cid not in fused:
            fused[cid] = dict(chunk)
            fused[cid]["rrf_score"]  = 0.0
            fused[cid]["bm25_rank"]  = -1
            fused[cid]["dense_rank"] = -1
        fused[cid]["rrf_score"] += _rrf_score(rank, rrf_k)
        fused[cid]["dense_rank"]  = rank

    # ── Sort by RRF score and return top_k ───────────────────
    merged = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)
    return merged[:top_k]
