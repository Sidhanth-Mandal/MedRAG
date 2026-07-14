"""
pipeline/qdrant_uploader.py
─────────────────────────────
Manages the Qdrant Cloud collection and upserts chunk vectors.

Collection schema:
  - Vector: 768-dim Cosine (bge-base-en-v1.5)
  - Payload: all chunk metadata fields
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import cfg

load_dotenv()

_QDRANT_URL     = os.environ.get("QDRANT_URL",     "")
_QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")


# ── Client factory ────────────────────────────────────────────────────────────

def get_qdrant_client() -> QdrantClient:
    if not _QDRANT_URL:
        raise ValueError("QDRANT_URL not set in .env")
    if not _QDRANT_API_KEY:
        raise ValueError("QDRANT_API_KEY not set in .env")

    return QdrantClient(
        url=_QDRANT_URL,
        api_key=_QDRANT_API_KEY,
        timeout=60,
    )


# ── Collection setup ──────────────────────────────────────────────────────────

def ensure_collection(
    client: QdrantClient,
    collection_name: str = None,
    force_recreate: bool = False,
) -> None:
    """
    Create the Qdrant collection if it doesn't exist.
    Set force_recreate=True to wipe and rebuild from scratch.
    """
    name = collection_name or cfg.qdrant.collection_name
    dim  = cfg.embedding.dimension

    existing = [c.name for c in client.get_collections().collections]

    if name in existing:
        if force_recreate:
            print(f"  [INFO] Deleting existing collection '{name}' …")
            client.delete_collection(name)
        else:
            print(f"  [INFO] Collection '{name}' already exists — skipping create.")
            return

    print(f"  [INFO] Creating collection '{name}' (dim={dim}, metric=Cosine) …")
    client.create_collection(
        collection_name=name,
        vectors_config=qmodels.VectorParams(
            size=dim,
            distance=qmodels.Distance.COSINE,
        ),
    )

    # Create payload indexes for fast filtered search
    for field, schema_type in [
        ("source_type", qmodels.PayloadSchemaType.KEYWORD),
        ("condition",   qmodels.PayloadSchemaType.KEYWORD),
        ("doc_id",      qmodels.PayloadSchemaType.KEYWORD),
    ]:
        client.create_payload_index(
            collection_name=name,
            field_name=field,
            field_schema=schema_type,
        )

    print(f"  [OK] Collection '{name}' created with payload indexes.")


# ── Upsert helpers ────────────────────────────────────────────────────────────

def _chunk_id_to_uuid(chunk_id: str) -> str:
    """Deterministically convert chunk_id string to UUID v5."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def upsert_chunks(
    client: QdrantClient,
    chunks: list[dict[str, Any]],
    vectors: list[list[float]],
    collection_name: str = None,
    batch_size: int = 100,
) -> None:
    """
    Upsert chunk records + their embedding vectors into Qdrant.

    Args:
        client:          QdrantClient instance
        chunks:          list of chunk dicts (from chunker.py)
        vectors:         parallel list of embedding vectors
        collection_name: override collection from config
        batch_size:      how many points to upsert per API call
    """
    name = collection_name or cfg.qdrant.collection_name

    assert len(chunks) == len(vectors), (
        f"Mismatch: {len(chunks)} chunks vs {len(vectors)} vectors"
    )

    points: list[qmodels.PointStruct] = []

    for chunk, vector in zip(chunks, vectors):
        # Build a clean payload dict (no large nested objects)
        payload = {
            "chunk_id":    chunk["chunk_id"],
            "doc_id":      chunk["doc_id"],
            "chunk_index": chunk["chunk_index"],
            "chunk_text":  chunk["chunk_text"],
            "token_count": chunk["token_count"],
            "title":       chunk["title"],
            "authors":     chunk["authors"][:5],   # cap to 5 for storage
            "journal":     chunk["journal"],
            "pub_date":    chunk["pub_date"],
            "url":         chunk["url"],
            "source_type": chunk["source_type"],
            "condition":   chunk["condition"],
            "pmid":        chunk.get("pmid"),
        }

        points.append(
            qmodels.PointStruct(
                id=_chunk_id_to_uuid(chunk["chunk_id"]),
                vector=vector,
                payload=payload,
            )
        )

    # Upsert in batches
    total_upserted = 0
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=name, points=batch, wait=True)
        total_upserted += len(batch)
        print(f"  Upserted {total_upserted}/{len(points)} points …", end="\r")

    print(f"\n  [OK] Upserted {total_upserted} points into '{name}'")


# ── Convenience search for verification ──────────────────────────────────────

def search(
    client: QdrantClient,
    query_vector: list[float],
    top_k: int = 5,
    source_type: str | None = None,
    condition: str | None = None,
    collection_name: str = None,
) -> list[dict[str, Any]]:
    """
    Dense vector search with optional payload filters.
    Returns list of payload dicts with added 'score' field.
    """
    name = collection_name or cfg.qdrant.collection_name

    # Build filter conditions
    must_conditions = []
    if source_type:
        must_conditions.append(
            qmodels.FieldCondition(
                key="source_type",
                match=qmodels.MatchValue(value=source_type),
            )
        )
    if condition:
        must_conditions.append(
            qmodels.FieldCondition(
                key="condition",
                match=qmodels.MatchValue(value=condition),
            )
        )

    query_filter = (
        qmodels.Filter(must=must_conditions) if must_conditions else None
    )

    # qdrant-client >= 1.9 replaced client.search() with client.query_points()
    response = client.query_points(
        collection_name=name,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    )

    return [
        {**hit.payload, "score": hit.score}
        for hit in response.points
    ]
