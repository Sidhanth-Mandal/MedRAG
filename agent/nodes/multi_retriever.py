"""
agent/nodes/multi_retriever.py
────────────────────────────────
Multi-query retriever node: executes all sub-queries from the
query_rewriter in sequence, merges results, deduplicates by chunk_id,
then reranks the merged pool against the ORIGINAL user query.

This ensures that for complex questions (comparisons, multi-condition,
multi-drug), each angle of the question gets independent retrieval
coverage, while the final reranking produces a single coherent list
ordered by relevance to the full question.

Falls back to a single-query retrieval if search_queries is empty.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from agent.state import AgentState
from config import cfg
from retrieval import hybrid_retriever, reranker


def multi_retriever_node(state: AgentState) -> dict:
    """
    Fan-out retrieval over multiple sub-queries, merge, deduplicate, rerank.
    """
    queries     = state.get("search_queries") or [state.get("search_query") or state["query"]]
    original_q  = state["query"]        # used as the rerank anchor
    route       = state.get("route", "both")
    source_type = None if route == "both" else route

    all_candidates: list[dict] = []

    for q in queries:
        try:
            results = hybrid_retriever.search(
                query=q,
                source_type=source_type,
                top_k=cfg.retrieval.rerank_top_k,
            )
            all_candidates.extend(results)
            print(f"[multi_retriever] sub-query '{q[:60]}' → {len(results)} candidates")
        except Exception as exc:
            print(f"[WARN] multi_retriever sub-query failed '{q[:60]}': {exc}")

    if not all_candidates:
        return {"retrieved_docs": []}

    # Deduplicate by chunk_id — keep highest rerank_score copy
    seen: dict[str, dict] = {}
    for doc in all_candidates:
        cid   = doc.get("chunk_id") or doc.get("doc_id") or id(doc)
        score = doc.get("rerank_score", 0.0)
        if cid not in seen or score > seen[cid].get("rerank_score", 0.0):
            seen[cid] = doc
    merged = list(seen.values())

    print(f"[multi_retriever] {len(all_candidates)} raw → {len(merged)} after dedup")

    # Rerank merged pool using the ORIGINAL user query for holistic ordering
    try:
        reranked_docs = reranker.rerank(
            query=original_q,
            chunks=merged,
            top_k=cfg.retrieval.final_top_k,
        )
    except Exception as exc:
        print(f"[WARN] multi_retriever reranking failed: {exc} — returning merged top-k")
        reranked_docs = merged[: cfg.retrieval.final_top_k]

    print(f"[multi_retriever] final docs after rerank: {len(reranked_docs)}")
    return {"retrieved_docs": reranked_docs}
