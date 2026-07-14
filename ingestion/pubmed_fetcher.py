"""
ingestion/pubmed_fetcher.py
────────────────────────────
Pulls PubMed abstracts via Biopython Entrez for each condition
defined in config.py, returning a list of structured dicts.

Usage (direct):
    python ingestion/pubmed_fetcher.py
"""

from __future__ import annotations

import os
import sys
import time
import json
from pathlib import Path
from typing import Any

from Bio import Entrez
from dotenv import load_dotenv
from tqdm import tqdm

# Allow running from project root or from ingestion/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import cfg

load_dotenv()

# NCBI requires an email for Entrez API access
Entrez.email = os.environ.get("ENTREZ_EMAIL", "your-email@example.com")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _search_pubmed(query: str, max_results: int, date_start: str, date_end: str) -> list[str]:
    """Return a list of PMIDs matching the query within the date range."""
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=max_results,
        mindate=date_start,
        maxdate=date_end,
        datetype="pdat",
        sort="relevance",
    )
    record = Entrez.read(handle)
    handle.close()
    return record.get("IdList", [])


def _fetch_abstracts(pmids: list[str], batch_size: int = 50) -> list[dict[str, Any]]:
    """Fetch full abstract records for the given PMIDs in batches."""
    records: list[dict[str, Any]] = []

    for i in tqdm(range(0, len(pmids), batch_size), desc="  Fetching batches"):
        batch = pmids[i : i + batch_size]
        ids_str = ",".join(batch)

        try:
            handle = Entrez.efetch(
                db="pubmed",
                id=ids_str,
                rettype="xml",
                retmode="xml",
            )
            batch_records = Entrez.read(handle)
            handle.close()
        except Exception as exc:
            print(f"  [WARN] Batch {i//batch_size + 1} failed: {exc}. Skipping.")
            time.sleep(2)
            continue

        for article in batch_records.get("PubmedArticle", []):
            parsed = _parse_article(article)
            if parsed:
                records.append(parsed)

        # Be polite to NCBI: max 3 requests/second without API key
        time.sleep(0.35)

    return records


def _parse_article(article: dict) -> dict[str, Any] | None:
    """Extract relevant fields from a raw PubMed XML record."""
    try:
        medline   = article["MedlineCitation"]
        art       = medline["Article"]
        pmid      = str(medline["PMID"])

        # Title
        title = str(art.get("ArticleTitle", "")).strip()

        # Abstract (may be structured with multiple sections)
        abstract_obj = art.get("Abstract", {})
        abstract_texts = abstract_obj.get("AbstractText", [])
        if isinstance(abstract_texts, list):
            abstract = " ".join(str(t) for t in abstract_texts).strip()
        else:
            abstract = str(abstract_texts).strip()

        # Skip records with no usable abstract
        if not abstract or len(abstract) < 50:
            return None

        # Authors
        author_list = art.get("AuthorList", [])
        authors = []
        for a in author_list:
            last  = a.get("LastName", "")
            first = a.get("ForeName", "")
            if last:
                authors.append(f"{last} {first}".strip())

        # Journal + date
        journal_info = art.get("Journal", {})
        journal_name = str(journal_info.get("Title", ""))
        pub_date_obj = (
            journal_info.get("JournalIssue", {}).get("PubDate", {})
        )
        year  = str(pub_date_obj.get("Year",  ""))
        month = str(pub_date_obj.get("Month", ""))
        pub_date = f"{year}-{month}" if month else year

        return {
            "id":          f"PMID_{pmid}",
            "pmid":        pmid,
            "title":       title,
            "text":        abstract,
            "authors":     authors,
            "journal":     journal_name,
            "pub_date":    pub_date,
            "url":         f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "source_type": "research",
            "condition":   None,   # filled in by caller
        }
    except Exception as exc:
        print(f"  [WARN] Parse error for article: {exc}")
        return None


# ── Public API ───────────────────────────────────────────────────────────────

def fetch_pubmed_for_condition(
    condition: str,
    query: str,
    max_results: int,
    date_start: str,
    date_end: str,
) -> list[dict[str, Any]]:
    """Fetch and return parsed PubMed abstracts for a single condition."""
    print(f"\n[SEARCH] Searching PubMed for '{condition}' …")
    print(f"   Query: {query}")

    pmids = _search_pubmed(query, max_results, date_start, date_end)
    print(f"   Found {len(pmids)} PMIDs -- fetching abstracts …")

    records = _fetch_abstracts(pmids)
    for r in records:
        r["condition"] = condition

    print(f"   [OK] {len(records)} abstracts with usable text for '{condition}'")
    return records


def fetch_all_pubmed() -> list[dict[str, Any]]:
    """Fetch PubMed abstracts for all conditions in config."""
    ic = cfg.ingestion
    all_records: list[dict[str, Any]] = []

    for condition in ic.conditions:
        query = ic.pubmed_queries[condition]
        records = fetch_pubmed_for_condition(
            condition=condition,
            query=query,
            max_results=ic.pubmed_max_per_condition,
            date_start=ic.pubmed_date_start,
            date_end=ic.pubmed_date_end,
        )
        all_records.extend(records)

    return all_records


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    records = fetch_all_pubmed()
    out_path = Path(cfg.ingestion.pubmed_output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"\n[SAVE] Saved {len(records)} records → {out_path}")

    # Quick preview
    if records:
        sample = records[0]
        print("\n── Sample record ──────────────────────────")
        print(f"  ID:        {sample['id']}")
        print(f"  Condition: {sample['condition']}")
        print(f"  Title:     {sample['title'][:80]}…")
        print(f"  Text:      {sample['text'][:120]}…")
        print(f"  Date:      {sample['pub_date']}")
        print(f"  Authors:   {', '.join(sample['authors'][:3])}")
