"""
eval/run_eval.py
─────────────────
Runs all 15 test cases through the full agent pipeline
and prints a pass/fail report with details.

Usage:
    python eval/run_eval.py
    python eval/run_eval.py --verbose    # show full answers
    python eval/run_eval.py --id T01     # run a single test by ID
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from agent.graph import run_agent, get_app
from eval.test_cases import TEST_CASES


def run_eval(verbose: bool = False, filter_id: str | None = None) -> None:
    print("\n" + "=" * 70)
    print("  AGENTIC RAG EVALUATION")
    print("=" * 70)

    cases = TEST_CASES
    if filter_id:
        cases = [c for c in cases if c["id"] == filter_id]
        if not cases:
            print(f"  [WARN] No test case with id '{filter_id}'")
            return

    print(f"\n  Loading agent graph...")
    app = get_app()
    print(f"  [OK] Agent ready. Running {len(cases)} test(s)...\n")

    results = []

    for case in cases:
        test_id  = case["id"]
        question = case["question"]
        expected = case["expected"]
        check_fn = case["check_fn"]

        print(f"  [{test_id}] {question[:65]}...")
        print(f"         Expected: {expected}")

        t_start = time.time()
        try:
            result  = run_agent(
                query=question,
                session_id=f"eval_{test_id}",
                app=app,
            )
            elapsed = time.time() - t_start

            passed, reason = check_fn(result)
            status = "PASS" if passed else "FAIL"

            print(f"         Result:   {status}  ({reason})  [{elapsed:.1f}s]")
            if verbose:
                print(f"         Route:    {result.get('route','?')}")
                print(f"         Refused:  {result.get('is_refused', False)}")
                print(f"         Conflict: {result.get('has_contradiction', False)}")
                print(f"         Docs:     {len(result.get('retrieved_docs', []))}")
                print(f"         Answer:   {result.get('answer','')[:200]}...")
            print()

            results.append({
                "id":       test_id,
                "question": question,
                "expected": expected,
                "passed":   passed,
                "reason":   reason,
                "elapsed":  elapsed,
                "result":   result,
            })

        except Exception as exc:
            elapsed = time.time() - t_start
            print(f"         Result:   ERROR  [{elapsed:.1f}s]")
            print(f"         Error:    {exc}")
            print()
            results.append({
                "id":       test_id,
                "question": question,
                "expected": expected,
                "passed":   False,
                "reason":   f"Exception: {exc}",
                "elapsed":  elapsed,
                "result":   {},
            })

    # ── Summary ────────────────────────────────────────────────────────────
    n_pass  = sum(1 for r in results if r["passed"])
    n_total = len(results)
    n_fail  = n_total - n_pass
    pct     = 100 * n_pass / n_total if n_total else 0
    total_t = sum(r["elapsed"] for r in results)

    print("=" * 70)
    print("  EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  Total tests:  {n_total}")
    print(f"  Passed:       {n_pass}  ({pct:.0f}%)")
    print(f"  Failed:       {n_fail}")
    print(f"  Total time:   {total_t:.1f}s")
    print()

    if n_fail > 0:
        print("  Failed tests:")
        for r in results:
            if not r["passed"]:
                print(f"    [{r['id']}] {r['question'][:55]}")
                print(f"           expected={r['expected']}")
                print(f"           reason={r['reason']}")

    print("=" * 70)

    # ── Per-category breakdown ──────────────────────────────────────────────
    categories: dict[str, list] = {}
    for r in results:
        cat = r["expected"]
        categories.setdefault(cat, []).append(r)

    print("\n  Breakdown by expected behavior:")
    for cat, cat_results in sorted(categories.items()):
        cat_pass = sum(1 for r in cat_results if r["passed"])
        print(f"    {cat:<30}  {cat_pass}/{len(cat_results)}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation suite")
    parser.add_argument("--verbose", action="store_true", help="Show full answers")
    parser.add_argument("--id",      type=str,            help="Run a single test by ID (e.g. T01)")
    args = parser.parse_args()

    run_eval(verbose=args.verbose, filter_id=args.id)
