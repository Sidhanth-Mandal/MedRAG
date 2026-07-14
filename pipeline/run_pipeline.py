"""
pipeline/run_pipeline.py
─────────────────────────
Step 2 orchestrator: loads raw JSON → chunks → embeds → upserts to Qdrant.

Usage:
    python pipeline/run_pipeline.py                  # full run
    python pipeline/run_pipeline.py --recreate       # wipe collection and rebuild
    python pipeline/run_pipeline.py --dry-run        # chunk only, no embedding/upload
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import cfg
from pipeline.chunker import chunk_all_documents
from pipeline.embedder import get_embedder
from pipeline.qdrant_uploader import get_qdrant_client, ensure_collection, upsert_chunks

load_dotenv()


def load_raw_data() -> list[dict]:
    """Load and merge PubMed + MedlinePlus JSON files."""
    pubmed_path = Path(cfg.ingestion.pubmed_output_path)
    mlp_path    = Path(cfg.ingestion.medlineplus_output_path)

    docs = []

    if pubmed_path.exists():
        pubmed_docs = json.loads(pubmed_path.read_text(encoding="utf-8"))
        docs.extend(pubmed_docs)
        print(f"  [OK] Loaded {len(pubmed_docs)} PubMed abstracts")
    else:
        print(f"  [WARN] {pubmed_path} not found — run ingestion first!")

    if mlp_path.exists():
        mlp_docs = json.loads(mlp_path.read_text(encoding="utf-8"))
        docs.extend(mlp_docs)
        print(f"  [OK] Loaded {len(mlp_docs)} MedlinePlus guidelines")
    else:
        print(f"  [WARN] {mlp_path} not found — run ingestion first!")

    return docs


def run_pipeline(recreate: bool = False, dry_run: bool = False) -> None:
    print("\n" + "=" * 60)
    print("  STEP 2: Chunking + Embedding + Qdrant Upload")
    print("=" * 60)

    # ── 1. Load raw data ──────────────────────────────────────
    print("\n[1/4] Loading raw data ...")
    docs = load_raw_data()
    print(f"       Total documents: {len(docs)}")

    # ── 2. Chunk all documents ────────────────────────────────
    print("\n[2/4] Chunking documents ...")
    t0 = time.time()
    all_chunks = chunk_all_documents(docs)
    elapsed = time.time() - t0

    print(f"       Total chunks:    {len(all_chunks)}")
    print(f"       Avg chunks/doc:  {len(all_chunks)/max(1,len(docs)):.1f}")
    print(f"       Time:            {elapsed:.1f}s")

    # Token stats
    token_counts = [c["token_count"] for c in all_chunks]
    print(f"       Avg tokens/chunk: {sum(token_counts)/len(token_counts):.0f}")
    print(f"       Max tokens/chunk: {max(token_counts)}")

    # Breakdown by source_type
    research_chunks  = [c for c in all_chunks if c["source_type"] == "research"]
    guideline_chunks = [c for c in all_chunks if c["source_type"] == "guideline"]
    print(f"       Research chunks:  {len(research_chunks)}")
    print(f"       Guideline chunks: {len(guideline_chunks)}")

    if dry_run:
        print("\n[DRY RUN] Skipping embedding and upload.")
        # Show a few sample chunks
        for i, chunk in enumerate(all_chunks[:3]):
            print(f"\n  Sample chunk {i+1}:")
            print(f"    chunk_id:    {chunk['chunk_id']}")
            print(f"    condition:   {chunk['condition']}")
            print(f"    source_type: {chunk['source_type']}")
            print(f"    tokens:      {chunk['token_count']}")
            print(f"    text:        {chunk['chunk_text'][:100]}...")
        return

    # ── 3. Embed all chunks ───────────────────────────────────
    print("\n[3/4] Embedding chunks via Cloudflare Workers AI ...")
    embedder    = get_embedder()
    chunk_texts = [c["chunk_text"] for c in all_chunks]

    all_vectors: list[list[float]] = []
    batch_size  = cfg.embedding.batch_size

    pbar = tqdm(
        range(0, len(chunk_texts), batch_size),
        desc="  Embedding batches",
        unit="batch",
    )

    for i in pbar:
        batch   = chunk_texts[i : i + batch_size]
        vectors = embedder.embed_batch(batch)
        all_vectors.extend(vectors)
        pbar.set_postfix({"done": len(all_vectors), "total": len(chunk_texts)})

    print(f"  [OK] {len(all_vectors)} vectors (dim={len(all_vectors[0])})")

    # ── 4. Upsert to Qdrant ───────────────────────────────────
    print("\n[4/4] Uploading to Qdrant Cloud ...")
    client = get_qdrant_client()
    ensure_collection(client, force_recreate=recreate)
    upsert_chunks(client, all_chunks, all_vectors)

    # Final stats (qdrant-client >= 1.9: vectors_count may be None until fully indexed)
    try:
        collection_info = client.get_collection(cfg.qdrant.collection_name)
        points = getattr(collection_info, "points_count", None) or \
                 getattr(collection_info, "vectors_count", "?")
        print(f"\n  [OK] Qdrant collection '{cfg.qdrant.collection_name}':")
        print(f"       Points indexed: {points}")
    except Exception:
        print(f"\n  [OK] Upsert complete (could not fetch collection stats).")
    print("\n  Step 2 complete! Run verify_search.py to check results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2: chunk, embed, upload to Qdrant")
    parser.add_argument("--recreate",  action="store_true", help="Wipe and rebuild Qdrant collection")
    parser.add_argument("--dry-run",   action="store_true", help="Chunk only, skip embedding/upload")
    args = parser.parse_args()

    run_pipeline(recreate=args.recreate, dry_run=args.dry_run)
