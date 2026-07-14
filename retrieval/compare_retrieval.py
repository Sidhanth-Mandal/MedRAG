"""
retrieval/compare_retrieval.py
────────────────────────────────
Side-by-side comparison of:
  1. Dense-only retrieval
  2. Hybrid (BM25 + dense) with RRF
  3. Hybrid + Cohere reranking

Run:
    python retrieval/compare_retrieval.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

from dotenv import load_dotenv

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from retrieval import dense_retriever, hybrid_retriever, reranker

load_dotenv()

TEST_QUERIES = [
    {
        "query":       "metformin efficacy HbA1c type 2 diabetes first line",
        "source_type": "research",
        "label":       "Diabetes: drug efficacy",
    },
    {
        "query":       "blood pressure target goals hypertension elderly patients",
        "source_type": None,
        "label":       "Hypertension: treatment targets",
    },
    {
        "query":       "asthma inhaler technique spacer children adherence",
        "source_type": "guideline",
        "label":       "Asthma: guideline — inhaler use",
    },
]


def _print_results(label: str, results: list, score_key: str) -> None:
    print(f"  --- {label} ---")
    for i, r in enumerate(results[:5], 1):
        score = r.get(score_key, r.get("rrf_score", 0))
        print(f"  #{i} {score_key}={score:.4f}  [{r['source_type']:<10}]  "
              f"{r['title'][:55]}")
        print(f"       {r['chunk_text'][:100]}...")
    print()


def run_compare() -> None:
    print("\n" + "=" * 72)
    print("  RETRIEVAL COMPARISON")
    print("=" * 72)

    for test in TEST_QUERIES:
        query       = test["query"]
        source_type = test.get("source_type")

        print(f"\n[QUERY] {test['label']}")
        print(f"  Text:   \"{query}\"")
        if source_type:
            print(f"  Filter: source_type='{source_type}'")
        print()

        # 1. Dense only
        dense_results = dense_retriever.search(query, top_k=5, source_type=source_type)
        _print_results("Dense only", dense_results, "dense_score")

        # 2. Hybrid (BM25 + dense, RRF)
        hybrid_results = hybrid_retriever.search(query, source_type=source_type, top_k=30)
        _print_results("Hybrid (RRF)", hybrid_results, "rrf_score")

        # 3. Hybrid + Cohere rerank
        reranked = reranker.rerank(query, hybrid_results, top_k=5)
        _print_results("Hybrid + Rerank", reranked, "rerank_score")

        print("=" * 72)


if __name__ == "__main__":
    run_compare()
