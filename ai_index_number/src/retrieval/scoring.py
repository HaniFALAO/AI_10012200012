from __future__ import annotations

import re
from typing import Any

import numpy as np


def normalize_scores(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.array(list(values.values()), dtype=float)
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-9:
        return {k: 1.0 for k in values}
    return {k: (v - mn) / (mx - mn) for k, v in values.items()}


def classify_query(query: str) -> str:
    q = query.lower()
    election_terms = ["election", "votes", "constituency", "party", "candidate", "ballot"]
    budget_terms = ["budget", "fiscal", "revenue", "expenditure", "gdp", "tax", "deficit"]
    e = any(t in q for t in election_terms)
    b = any(t in q for t in budget_terms)
    if e and b:
        return "mixed"
    if e:
        return "election"
    if b:
        return "budget"
    return "mixed"


def keyword_overlap_bonus(query: str, text: str) -> float:
    q_words = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))
    t_words = set(re.findall(r"[a-zA-Z0-9]+", text.lower()))
    if not q_words:
        return 0.0
    return len(q_words.intersection(t_words)) / len(q_words)


def year_numeric_bonus(query: str, chunk: dict[str, Any]) -> float:
    years_in_q = set(re.findall(r"\b(19\d{2}|20\d{2}|21\d{2})\b", query))
    nums_in_q = re.findall(r"\b\d[\d,\.]*\b", query)
    bonus = 0.0
    if years_in_q and chunk.get("year") and str(chunk["year"]) in years_in_q:
        bonus += 0.2
    if nums_in_q and any(n in chunk["text"] for n in nums_in_q):
        bonus += 0.1
    return bonus


def source_match_bonus(query_type: str, chunk_source: str) -> float:
    if query_type == "election" and chunk_source == "election_csv":
        return 0.2
    if query_type == "budget" and chunk_source == "budget_pdf":
        return 0.2
    return 0.05 if query_type == "mixed" else 0.0
