"""
ingestion/run_ingestion.py
───────────────────────────
Orchestrates the full Step 1 data pipeline:
  1. Pull PubMed abstracts for all conditions
  2. Pull MedlinePlus guidelines for all conditions
  3. Save both to data/raw/ as JSON
  4. Print a summary report

Usage:
    python ingestion/run_ingestion.py [--pubmed-only] [--medlineplus-only]
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

# Force UTF-8 output on Windows to avoid emoji encode errors
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import cfg
from ingestion.pubmed_fetcher import fetch_all_pubmed
from ingestion.medlineplus_fetcher import fetch_all_medlineplus

load_dotenv()


def save_json(data: list, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  [SAVE] Saved {len(data)} records → {out}")


def print_summary(pubmed: list, medlineplus: list) -> None:
    """Print a breakdown of records by condition and source type."""
    all_records = pubmed + medlineplus
    print("\n" + "=" * 60)
    print("  INGESTION SUMMARY")
    print("=" * 60)
    print(f"  Total records:        {len(all_records)}")
    print(f"  PubMed (research):    {len(pubmed)}")
    print(f"  MedlinePlus (guide):  {len(medlineplus)}")
    print()

    # Breakdown by condition
    from collections import Counter
    condition_counts = Counter(r["condition"] for r in all_records)
    for condition, count in sorted(condition_counts.items()):
        pubmed_cnt = sum(1 for r in pubmed if r["condition"] == condition)
        mlp_cnt    = sum(1 for r in medlineplus if r["condition"] == condition)
        print(f"  {condition:<22}  {count:>4} total  "
              f"({pubmed_cnt} research + {mlp_cnt} guideline)")

    # Field completeness checks
    print()
    print("  Field completeness:")
    for field in ["title", "text", "pub_date", "url", "condition"]:
        filled = sum(1 for r in all_records if r.get(field))
        pct    = 100 * filled / len(all_records) if all_records else 0
        print(f"    {field:<12} {filled:>4}/{len(all_records)}  ({pct:.0f}%)")

    print()
    avg_len = (
        sum(len(r["text"]) for r in all_records) / len(all_records)
        if all_records else 0
    )
    print(f"  Avg text length:      {avg_len:.0f} chars")
    print("=" * 60)

    # Timestamp
    print(f"\n  Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    parser = argparse.ArgumentParser(description="Run data ingestion for Agentic RAG")
    parser.add_argument("--pubmed-only",      action="store_true", help="Only fetch PubMed")
    parser.add_argument("--medlineplus-only", action="store_true", help="Only fetch MedlinePlus")
    args = parser.parse_args()

    pubmed_records:     list = []
    medlineplus_records: list = []

    print("\n[START] Starting data ingestion …\n")

    # ── PubMed ──────────────────────────────────────────────
    if not args.medlineplus_only:
        print("=" * 60)
        print("  STEP 1a: PubMed Abstracts")
        print("=" * 60)
        pubmed_records = fetch_all_pubmed()
        save_json(pubmed_records, cfg.ingestion.pubmed_output_path)

    # ── MedlinePlus ─────────────────────────────────────────
    if not args.pubmed_only:
        print("\n" + "=" * 60)
        print("  STEP 1b: MedlinePlus Guidelines")
        print("=" * 60)
        medlineplus_records = fetch_all_medlineplus()
        save_json(medlineplus_records, cfg.ingestion.medlineplus_output_path)

    # ── Summary ─────────────────────────────────────────────
    if pubmed_records or medlineplus_records:
        print_summary(pubmed_records, medlineplus_records)
    else:
        print("[WARN]  No records fetched. Check your .env and network connection.")


if __name__ == "__main__":
    main()
