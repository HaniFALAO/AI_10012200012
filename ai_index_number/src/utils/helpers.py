from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_years(text: str) -> list[int]:
    years = re.findall(r"\b(19\d{2}|20\d{2}|21\d{2})\b", text)
    return sorted({int(y) for y in years})


def extract_numbers(text: str) -> list[str]:
    return re.findall(r"\b\d[\d,\.]*\b", text)


def text_keywords(text: str, max_keywords: int = 12) -> list[str]:
    stop = {
        "the",
        "and",
        "of",
        "to",
        "in",
        "for",
        "on",
        "with",
        "is",
        "are",
        "a",
        "an",
        "at",
        "from",
        "by",
        "as",
        "that",
        "this",
        "be",
        "or",
    }
    tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
    filtered = [t for t in tokens if t not in stop]
    freq: dict[str, int] = {}
    for tok in filtered:
        freq[tok] = freq.get(tok, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in ranked[:max_keywords]]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")
