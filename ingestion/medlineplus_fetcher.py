"""
ingestion/medlineplus_fetcher.py
─────────────────────────────────
Pulls patient-friendly guideline content from MedlinePlus
using the NLM Web Services Search API (no scraping needed).

Endpoint:
  https://wsearch.nlm.nih.gov/ws/query?db=healthTopics&term=<query>

Returns structured XML with health topic summaries, full-text
sections (overview, symptoms, treatment, etc.), and stable URLs.

Usage (direct):
    python ingestion/medlineplus_fetcher.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import requests
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import cfg

load_dotenv()

# MedlinePlus Health Topics search endpoint
_BASE_URL = "https://wsearch.nlm.nih.gov/ws/query"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _fetch_health_topics(query: str, max_results: int = 20) -> list[dict[str, Any]]:
    """Query the MedlinePlus Web Services API and return parsed topic dicts."""
    params = {
        "db":      "healthTopics",
        "term":    query,
        "retmax":  max_results,
        "rettype": "brief",
    }

    resp = requests.get(_BASE_URL, params=params, timeout=30)
    resp.raise_for_status()

    return _parse_xml_response(resp.text, query)


def _parse_xml_response(xml_text: str, query: str) -> list[dict[str, Any]]:
    """Parse the MedlinePlus XML response into structured dicts."""
    records: list[dict[str, Any]] = []

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        print(f"  [WARN] XML parse error: {exc}")
        return records

    # Each health topic is a <document> element
    for doc in root.findall(".//document"):
        url = doc.attrib.get("url", "")

        # Extract named content sections
        fields: dict[str, str] = {}
        for content_el in doc.findall("content"):
            name  = content_el.attrib.get("name", "")
            value = (content_el.text or "").strip()
            # Strip embedded HTML tags for clean text
            value = _strip_html(value)
            if name and value:
                fields[name] = value

        title   = fields.get("title", "")
        summary = fields.get("FullSummary", "") or fields.get("snippet", "")

        # Combine all useful text sections into one document body
        section_keys = [
            "FullSummary", "mesh", "alt-title",
        ]
        body_parts = [fields.get(k, "") for k in section_keys if fields.get(k)]

        # Also collect any "section" elements (treatment, symptoms, etc.)
        all_text_parts = [summary]
        for key, val in fields.items():
            if key not in ("title", "FullSummary", "snippet") and val:
                all_text_parts.append(val)

        full_text = " ".join(filter(None, all_text_parts)).strip()

        if not title or not full_text or len(full_text) < 80:
            continue

        record_id = "MLP_" + url.rstrip("/").split("/")[-1].replace("#", "_")

        records.append({
            "id":          record_id,
            "pmid":        None,
            "title":       title,
            "text":        full_text,
            "authors":     ["MedlinePlus / NLM"],
            "journal":     "MedlinePlus",
            "pub_date":    "",          # MedlinePlus pages are continuously updated
            "url":         url,
            "source_type": "guideline",
            "condition":   None,        # filled in by caller
        })

    return records


def _strip_html(text: str) -> str:
    """Remove HTML tags from a string using the stdlib XML parser."""
    try:
        return ET.fromstring(f"<root>{text}</root>").itertext().__next__() or \
               "".join(ET.fromstring(f"<root>{text}</root>").itertext())
    except ET.ParseError:
        # Fallback: naive tag stripper
        import re
        return re.sub(r"<[^>]+>", " ", text).strip()


def _strip_html(text: str) -> str:
    """Remove HTML tags using a simple regex fallback (robust for messy HTML)."""
    import re
    # Remove HTML tags
    clean = re.sub(r"<[^>]+>", " ", text)
    # Collapse whitespace
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_medlineplus_for_condition(
    condition: str,
    query: str,
    max_results: int = 20,
) -> list[dict[str, Any]]:
    """Fetch MedlinePlus health topics for a single condition."""
    print(f"\n[GUIDE] Fetching MedlinePlus for '{condition}' …")
    print(f"   Query: {query}")

    records = _fetch_health_topics(query, max_results)
    for r in records:
        r["condition"] = condition

    print(f"   [OK] {len(records)} guideline documents for '{condition}'")
    time.sleep(0.5)
    return records


def fetch_all_medlineplus() -> list[dict[str, Any]]:
    """Fetch MedlinePlus content for all conditions in config."""
    ic = cfg.ingestion
    all_records: list[dict[str, Any]] = []

    for condition in ic.conditions:
        query = ic.medlineplus_queries[condition]
        records = fetch_medlineplus_for_condition(
            condition=condition,
            query=query,
            max_results=25,   # MedlinePlus has fewer topics per condition
        )
        all_records.extend(records)

    # Deduplicate by URL (same topic might appear for multiple queries)
    seen_urls: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for r in all_records:
        if r["url"] not in seen_urls:
            seen_urls.add(r["url"])
            deduped.append(r)

    print(f"\n[DEDUP] After dedup: {len(deduped)} unique MedlinePlus documents")
    return deduped


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    records = fetch_all_medlineplus()
    out_path = Path(cfg.ingestion.medlineplus_output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"\n[SAVE] Saved {len(records)} records → {out_path}")

    if records:
        sample = records[0]
        print("\n── Sample record ──────────────────────────")
        print(f"  ID:        {sample['id']}")
        print(f"  Condition: {sample['condition']}")
        print(f"  Title:     {sample['title']}")
        print(f"  URL:       {sample['url']}")
        print(f"  Text:      {sample['text'][:200]}…")
