from __future__ import annotations

from pathlib import Path

import fitz

from src.utils.helpers import normalize_whitespace


def load_budget_pdf(pdf_path: str) -> list[dict]:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    pages: list[dict] = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc):
            text = normalize_whitespace(page.get_text("text"))
            pages.append({"page_number": i + 1, "text": text})
    return pages
