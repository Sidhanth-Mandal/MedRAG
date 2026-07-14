"""
agent/nodes/retriever.py
─────────────────────────
Retriever node: runs hybrid search (BM25 + dense, RRF)
scoped to the route from the router node, then reranks.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from agent.state import AgentState
from config import cfg
from retrieval import hybrid_retriever, reranker


def retriever_node(state: AgentState) -> dict:
    """
    Retrieve relevant chunks using hybrid search + Cohere reranking.
    Uses search_query (which may be rewritten by grader) and route.
    """
    query       = state.get("search_query") or state["query"]
    route       = state.get("route", "both")

    # Map route → source_type filter
    source_type = None if route == "both" else route

    # Hybrid retrieval (BM25 + dense → RRF)
    hybrid_results = hybrid_retriever.search(
        query=query,
        source_type=source_type,
        top_k=cfg.retrieval.rerank_top_k,
    )

    # Cohere reranking → final top-k
    reranked_docs = reranker.rerank(
        query=query,
        chunks=hybrid_results,
        top_k=cfg.retrieval.final_top_k,
    )

    return {"retrieved_docs": reranked_docs}
