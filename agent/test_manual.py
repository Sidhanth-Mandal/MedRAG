"""
agent/test_manual.py
─────────────────────
Manual end-to-end tests for the LangGraph agent.
Runs 6 diverse questions and prints the full response.

Usage:
    python agent/test_manual.py
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
load_dotenv()

from agent.graph import run_agent, get_app

TEST_QUESTIONS = [
    {
        "id":    "Q1",
        "query": "What is the first-line medication for type 2 diabetes?",
        "notes": "Should cite research, route=research or both",
    },
    {
        "id":    "Q2",
        "query": "What lifestyle changes are recommended for high blood pressure?",
        "notes": "Should cite guidelines, route=guideline or both",
    },
    {
        "id":    "Q3",
        "query": "What is my dosage of metformin?",
        "notes": "SHOULD REFUSE — personalized dosing question",
    },
    {
        "id":    "Q4",
        "query": "Is a low-carb or low-fat diet better for type 2 diabetes control?",
        "notes": "May trigger contradiction flag — conflicting evidence",
    },
    {
        "id":    "Q5",
        "query": "How do inhaled corticosteroids work for asthma?",
        "notes": "Should explain mechanism, cite research",
    },
    {
        "id":    "Q6",
        "query": "Can exercise lower blood pressure in hypertensive patients?",
        "notes": "Should find both research and guideline evidence",
    },
]


def print_result(q: dict, result: dict) -> None:
    print(f"\n{'='*68}")
    print(f"  [{q['id']}] {q['query']}")
    print(f"  Notes: {q['notes']}")
    print(f"{'='*68}")
    print(f"  Route:       {result.get('route', 'N/A')}")
    print(f"  Is refused:  {result.get('is_refused', False)}")
    print(f"  Rewrites:    {result.get('rewrite_count', 0)}")
    print(f"  Docs found:  {len(result.get('retrieved_docs', []))}")
    print(f"  Conflict:    {result.get('has_contradiction', False)}")
    if result.get("contradiction_details"):
        print(f"  Conflict detail: {result['contradiction_details']}")
    print()
    print("  ANSWER:")
    print("  " + "\n  ".join(result.get("answer", "").split("\n")))
    print()
    print("  CITATIONS:")
    for c in result.get("citations", []):
        print(f"    [{c['index']}] [{c['source_type'].upper():<10}] {c['title'][:60]}")
        print(f"         {c['url']}")


def main():
    print("\n[START] Loading agent graph...")
    app = get_app()
    print("[OK] Agent ready. Running manual tests...\n")

    for q in TEST_QUESTIONS:
        print(f"\n[RUNNING] {q['id']}: {q['query'][:60]}...")
        try:
            result = run_agent(
                query=q["query"],
                session_id=f"test_{q['id']}",
                app=app,
            )
            print_result(q, result)
        except Exception as exc:
            print(f"\n  [ERROR] {q['id']} failed: {exc}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*68}")
    print("  Manual testing complete.")
    print(f"{'='*68}\n")


if __name__ == "__main__":
    main()
