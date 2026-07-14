"""
agent/nodes/safety.py
──────────────────────
Safety node: detects questions that ask for personalized
medical diagnosis or specific dosing advice, and refuses
them with a 'consult a doctor' message.

Uses a keyword pre-filter first (fast path), then falls
back to an LLM classifier for ambiguous cases.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from langchain_core.messages import AIMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from agent.llm import get_llm
from agent.state import AgentState

# ── Keyword fast-path ─────────────────────────────────────────────────────────

_UNSAFE_PATTERNS = [
    r"\bmy (doctor|physician|patient|medication|prescription|dose|dosage)\b",
    r"\b(diagnose|diagnos(e|is|ing)) me\b",
    r"\bam i (sick|diabetic|hypertensive|asthmatic|ill)\b",
    r"\b(what|which) (dose|dosage|mg|milligram|pill|tablet)\b",
    r"\bshould i (take|stop|start|increase|decrease)\b",
    r"\b(is it safe|safe for me) to\b",
    r"\bi have (diabetes|hypertension|asthma|high blood pressure)\b",
    r"\bmy (blood sugar|blood pressure|a1c|hba1c|fev1|peak flow)\b",
    r"\b(personal|personalized) (advice|recommendation|treatment|plan)\b",
    r"\bwhat (medication|drug|medicine) should i (take|use)\b",
    r"\bcan i (take|use|stop) (my )?\w+ (with|for|instead)\b",
    r"\b(prescribe|prescription) for me\b",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _UNSAFE_PATTERNS]

_REFUSE_MESSAGE = (
    "I'm not able to provide personalized medical advice, diagnosis, "
    "or specific dosing recommendations. This system provides general "
    "medical information for educational purposes only.\n\n"
    "Please consult a qualified healthcare professional (your doctor, "
    "pharmacist, or specialist) for advice tailored to your specific "
    "situation."
)

_SAFETY_PROMPT = """
You are a medical content safety classifier.

Question: {query}

Does this question ask for:
- Personalized diagnosis (diagnosing the user's specific condition)
- Specific dosing/medication advice for the user
- Whether the user should take/stop a specific drug
- Whether the user has a specific disease

Answer with ONLY one word: SAFE or UNSAFE.

Rule: Educational/factual questions about conditions, treatments, or
research findings in general are SAFE. Questions asking for personal
medical advice are UNSAFE.
"""


def safety_node(state: AgentState) -> dict:
    """Check if the query is safe to answer."""
    query = state["query"]

    # Fast path: keyword matching
    for pattern in _COMPILED:
        if pattern.search(query):
            return {
                "is_safe":     False,
                "is_refused":  True,
                "refuse_reason": "personalized_medical_advice",
                "answer":      _REFUSE_MESSAGE,
                "citations":   [],
            }

    # Slow path: LLM classifier for ambiguous cases
    try:
        llm    = get_llm(temperature=0.0, max_tokens=5)
        prompt = _SAFETY_PROMPT.format(query=query)
        result = llm.invoke(prompt)
        verdict = result.content.strip().upper()

        if "UNSAFE" in verdict:
            return {
                "is_safe":     False,
                "is_refused":  True,
                "refuse_reason": "personalized_medical_advice",
                "answer":      _REFUSE_MESSAGE,
                "citations":   [],
            }
    except Exception:
        # If LLM fails, default to safe (don't block legitimate queries)
        pass

    return {
        "is_safe":    True,
        "is_refused": False,
    }
