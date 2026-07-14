"""
pipeline/verify_search.py
──────────────────────────
Smoke-tests the Qdrant collection with dense vector search.
Runs 3 representative queries and prints the top-5 results
for each, so you can visually verify the retrieval makes sense.

Usage:
    python pipeline/verify_search.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

from dotenv import load_dotenv

# UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.embedder import get_embedder
from pipeline.qdrant_uploader import get_qdrant_client, search

load_dotenv()

TEST_QUERIES = [
    {
        "query":       "metformin first-line treatment type 2 diabetes HbA1c reduction",
        "source_type": "research",
        "description": "Research: diabetes drug efficacy",
    },
    {
        "query":       "lifestyle changes blood pressure hypertension diet exercise",
        "source_type": "guideline",
        "description": "Guideline: hypertension lifestyle advice",
    },
    {
        "query":       "inhaled corticosteroids asthma management exacerbation prevention",
        "source_type": None,
        "description": "Both: asthma corticosteroid treatment",
    },
]


def run_verify():
    print("\n" + "=" * 70)
    print("  STEP 2 VERIFICATION — Dense Vector Search")
    print("=" * 70)

    embedder = get_embedder()
    client   = get_qdrant_client()

    for i, test in enumerate(TEST_QUERIES, 1):
        print(f"\n[Query {i}] {test['description']}")
        print(f"  Text:        \"{test['query']}\"")
        if test["source_type"]:
            print(f"  Filter:      source_type='{test['source_type']}'")
        print()

        # Embed the query
        qvec = embedder.embed_one(test["query"])

        # Search Qdrant
        results = search(
            client,
            query_vector=qvec,
            top_k=5,
            source_type=test.get("source_type"),
        )

        if not results:
            print("  [WARN] No results returned — is the collection populated?")
            continue

        for rank, hit in enumerate(results, 1):
            print(f"  #{rank}  score={hit['score']:.4f}  "
                  f"[{hit['source_type'].upper():<10}]  "
                  f"condition={hit['condition']}")
            print(f"       title:  {hit['title'][:70]}")
            print(f"       text:   {hit['chunk_text'][:120]}...")
            if hit.get("pmid"):
                print(f"       url:    https://pubmed.ncbi.nlm.nih.gov/{hit['pmid']}/")
            else:
                print(f"       url:    {hit['url']}")
            print()

    print("=" * 70)
    print("  Verification complete. Check results above look medically sensible.")
    print("=" * 70)


if __name__ == "__main__":
    run_verify()
