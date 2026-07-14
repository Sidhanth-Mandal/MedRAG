"""
retrieval/bm25_retriever.py
────────────────────────────
Builds and queries a BM25 sparse-retrieval index over all chunks.

The index is built from the raw JSON files (not Qdrant) so it works
offline. It is serialized to data/bm25_index.pkl for fast reload.
"""

from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import cfg
from pipeline.chunker import chunk_all_documents


# ── Text pre-processing ──────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase, remove punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


# ── Index build / save / load ─────────────────────────────────────────────────

def build_index() -> tuple[BM25Okapi, list[dict[str, Any]]]:
    """
    Load all raw documents, chunk them, and build a BM25 index.
    Returns (bm25_model, chunks_list) — the chunks list is needed
    to map BM25 result indices back to chunk metadata.
    """
    pubmed_path = Path(cfg.ingestion.pubmed_output_path)
    mlp_path    = Path(cfg.ingestion.medlineplus_output_path)

    docs: list[dict] = []
    if pubmed_path.exists():
        docs.extend(json.loads(pubmed_path.read_text(encoding="utf-8")))
    if mlp_path.exists():
        docs.extend(json.loads(mlp_path.read_text(encoding="utf-8")))

    if not docs:
        raise FileNotFoundError(
            "No raw data found. Run ingestion/run_ingestion.py first."
        )

    print(f"  [BM25] Chunking {len(docs)} documents for BM25 index...")
    chunks = chunk_all_documents(docs)
    print(f"  [BM25] Building index over {len(chunks)} chunks...")

    tokenized_corpus = [_tokenize(c["chunk_text"]) for c in chunks]
    bm25 = BM25Okapi(
        tokenized_corpus,
        k1=cfg.bm25.k1,
        b=cfg.bm25.b,
    )

    return bm25, chunks


def save_index(bm25: BM25Okapi, chunks: list[dict[str, Any]]) -> None:
    """Serialize the BM25 model and chunk list to disk."""
    index_path = Path(cfg.bm25.index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)
    print(f"  [BM25] Index saved to {index_path}")


def load_index() -> tuple[BM25Okapi, list[dict[str, Any]]]:
    """Load a previously built BM25 index from disk."""
    index_path = Path(cfg.bm25.index_path)
    if not index_path.exists():
        print("  [BM25] Index not found on disk — building now...")
        bm25, chunks = build_index()
        save_index(bm25, chunks)
        return bm25, chunks

    print(f"  [BM25] Loading index from {index_path}...")
    with open(index_path, "rb") as f:
        data = pickle.load(f)
    print(f"  [BM25] Loaded index with {len(data['chunks'])} chunks")
    return data["bm25"], data["chunks"]


# ── Search ────────────────────────────────────────────────────────────────────

def search(
    query: str,
    bm25: BM25Okapi,
    chunks: list[dict[str, Any]],
    top_k: int = None,
    source_type: str | None = None,
    condition: str | None = None,
) -> list[dict[str, Any]]:
    """
    BM25 search. Returns top_k chunks sorted by BM25 score (descending).
    Each result has a 'bm25_score' field added.
    """
    top_k = top_k or cfg.bm25.top_k
    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    # Pair scores with chunk indices, filter, sort
    scored = [
        (score, idx)
        for idx, score in enumerate(scores)
        if score > 0
        and (source_type is None or chunks[idx]["source_type"] == source_type)
        and (condition  is None or chunks[idx]["condition"]   == condition)
    ]
    scored.sort(key=lambda x: x[0], reverse=True)

    results: list[dict[str, Any]] = []
    for score, idx in scored[:top_k]:
        chunk = dict(chunks[idx])
        chunk["bm25_score"] = float(score)
        results.append(chunk)

    return results


# ── Singleton loader ──────────────────────────────────────────────────────────

_bm25_index: tuple[BM25Okapi, list[dict[str, Any]]] | None = None


def get_bm25_index() -> tuple[BM25Okapi, list[dict[str, Any]]]:
    """Return the cached BM25 index, loading/building if needed."""
    global _bm25_index
    if _bm25_index is None:
        _bm25_index = load_index()
    return _bm25_index


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    bm25, chunks = build_index()
    save_index(bm25, chunks)

    query = "metformin blood sugar type 2 diabetes"
    results = search(query, bm25, chunks, top_k=5)
    print(f"\nBM25 top-5 for: '{query}'")
    for i, r in enumerate(results, 1):
        print(f"  #{i} score={r['bm25_score']:.3f}  [{r['source_type']}]  {r['title'][:60]}")
        print(f"       {r['chunk_text'][:100]}...")
