"""
pipeline/chunker.py
────────────────────
Splits raw documents (from data/raw/*.json) into overlapping
token-aware chunks, preserving all metadata on each chunk.

Each output chunk has:
  - All original doc fields (id, condition, source_type, etc.)
  - chunk_index  : position of this chunk within the doc
  - chunk_text   : the actual chunk content
  - chunk_id     : unique ID = "{doc_id}_chunk_{chunk_index}"
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import cfg


# ── Tokeniser ────────────────────────────────────────────────────────────────

_enc = tiktoken.get_encoding(cfg.chunk.tokenizer)


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _decode_tokens(tokens: list[int]) -> str:
    return _enc.decode(tokens)


# ── Core chunking logic ──────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> list[str]:
    """
    Split text into overlapping token-based chunks.
    Tries to break on sentence boundaries when possible.
    """
    chunk_size    = chunk_size    or cfg.chunk.chunk_size
    chunk_overlap = chunk_overlap or cfg.chunk.chunk_overlap

    tokens = _enc.encode(text)
    total  = len(tokens)

    if total <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < total:
        end = min(start + chunk_size, total)
        chunk_tokens = tokens[start:end]
        chunk_text_raw = _decode_tokens(chunk_tokens)

        # Try to trim to the last sentence boundary to avoid mid-sentence cuts
        if end < total:
            # Look for the last sentence-ending punctuation in the chunk
            match = re.search(r'[.!?]\s+', chunk_text_raw[::-1])
            if match and match.start() < chunk_size // 2:
                trim_chars = match.start()
                chunk_text_raw = chunk_text_raw[: len(chunk_text_raw) - trim_chars]

        chunks.append(chunk_text_raw.strip())

        if end == total:
            break

        # Move start forward by (chunk_size - overlap)
        start += max(1, chunk_size - chunk_overlap)

    return [c for c in chunks if c]  # drop empties


# ── Document-level chunking ──────────────────────────────────────────────────

def chunk_document(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Chunk a single document and return a list of chunk dicts."""
    text   = doc.get("text", "").strip()
    doc_id = doc["id"]

    if not text:
        return []

    text_chunks = chunk_text(text)
    result: list[dict[str, Any]] = []

    for i, chunk in enumerate(text_chunks):
        chunk_record = {
            # Inherited metadata
            "doc_id":      doc_id,
            "pmid":        doc.get("pmid"),
            "title":       doc.get("title", ""),
            "authors":     doc.get("authors", []),
            "journal":     doc.get("journal", ""),
            "pub_date":    doc.get("pub_date", ""),
            "url":         doc.get("url", ""),
            "source_type": doc.get("source_type", ""),
            "condition":   doc.get("condition", ""),
            # Chunk-specific fields
            "chunk_index": i,
            "chunk_text":  chunk,
            "chunk_id":    f"{doc_id}_chunk_{i}",
            "token_count": _count_tokens(chunk),
        }
        result.append(chunk_record)

    return result


def chunk_all_documents(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chunk all documents and return flat list of all chunk dicts."""
    all_chunks: list[dict[str, Any]] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    return all_chunks


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pubmed_path = Path(cfg.ingestion.pubmed_output_path)
    docs = json.loads(pubmed_path.read_text(encoding="utf-8"))[:5]  # first 5 docs

    for doc in docs:
        chunks = chunk_document(doc)
        print(f"\nDoc: {doc['id']} ({doc['condition']}) → {len(chunks)} chunks")
        for c in chunks:
            print(f"  [{c['chunk_index']}] tokens={c['token_count']}  "
                  f"text={c['chunk_text'][:60]}...")
