from __future__ import annotations

import re

from src.utils.helpers import normalize_whitespace


def clean_pdf_pages(pages: list[dict]) -> list[dict]:
    cleaned: list[dict] = []
    for page in pages:
        text = page["text"]
        text = re.sub(r"\s*Page\s+\d+\s*", " ", text, flags=re.IGNORECASE)
        text = normalize_whitespace(text)
        cleaned.append({"page_number": page["page_number"], "text": text})
    return cleaned
