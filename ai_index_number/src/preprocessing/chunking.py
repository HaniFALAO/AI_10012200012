from __future__ import annotations

import re
from typing import Literal

import pandas as pd

from src.utils.helpers import extract_years, normalize_whitespace, text_keywords

ChunkingMethod = Literal["fixed", "paragraph"]


def _word_windows(words: list[str], chunk_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        win = words[i : i + chunk_size]
        if len(win) < 40:
            continue
        chunks.append(" ".join(win))
        if i + chunk_size >= len(words):
            break
    return chunks


def election_rows_to_chunks(df: pd.DataFrame) -> list[dict]:
    chunks: list[dict] = []
    for idx, row in df.iterrows():
        row_text = " | ".join([f"{col}: {row[col]}" for col in df.columns])
        row_text = normalize_whitespace(row_text)
        years = extract_years(row_text)
        chunks.append(
            {
                "chunk_id": f"csv_{idx}",
                "source": "election_csv",
                "chunk_type": "record",
                "text": row_text,
                "section_title": "Election record",
                "year": years[0] if years else None,
                "keywords": text_keywords(row_text),
            }
        )
    return chunks


def pdf_fixed_chunks(pages: list[dict], chunk_size: int = 400, overlap: int = 80) -> list[dict]:
    chunks: list[dict] = []
    for page in pages:
        words = page["text"].split()
        windows = _word_windows(words, chunk_size=chunk_size, overlap=overlap)
        for j, text in enumerate(windows):
            years = extract_years(text)
            chunks.append(
                {
                    "chunk_id": f"pdf_fixed_p{page['page_number']}_{j}",
                    "source": "budget_pdf",
                    "chunk_type": "fixed",
                    "text": text,
                    "section_title": f"Page {page['page_number']}",
                    "year": years[0] if years else None,
                    "keywords": text_keywords(text),
                }
            )
    return chunks


def pdf_paragraph_chunks(pages: list[dict], target_min: int = 300, target_max: int = 500) -> list[dict]:
    chunks: list[dict] = []
    for page in pages:
        parts = [normalize_whitespace(p) for p in re.split(r"\n\s*\n|(?<=[\.!?])\s{2,}", page["text"]) if p.strip()]
        bucket: list[str] = []
        for para in parts:
            candidate = " ".join(bucket + [para]).strip()
            count = len(candidate.split())
            if count <= target_max:
                bucket.append(para)
                continue

            if bucket:
                text = " ".join(bucket)
                if len(text.split()) >= 80:
                    years = extract_years(text)
                    chunks.append(
                        {
                            "chunk_id": f"pdf_para_p{page['page_number']}_{len(chunks)}",
                            "source": "budget_pdf",
                            "chunk_type": "paragraph",
                            "text": text,
                            "section_title": f"Page {page['page_number']}",
                            "year": years[0] if years else None,
                            "keywords": text_keywords(text),
                        }
                    )
                overlap = " ".join(text.split()[-40:])
                bucket = [overlap, para]
            else:
                bucket = [para]

        if bucket:
            text = normalize_whitespace(" ".join(bucket))
            if len(text.split()) >= min(80, target_min // 3):
                years = extract_years(text)
                chunks.append(
                    {
                        "chunk_id": f"pdf_para_p{page['page_number']}_{len(chunks)}",
                        "source": "budget_pdf",
                        "chunk_type": "paragraph",
                        "text": text,
                        "section_title": f"Page {page['page_number']}",
                        "year": years[0] if years else None,
                        "keywords": text_keywords(text),
                    }
                )
    return chunks


def build_all_chunks(df: pd.DataFrame, cleaned_pages: list[dict], chunk_method: ChunkingMethod = "paragraph") -> list[dict]:
    csv_chunks = election_rows_to_chunks(df)
    if chunk_method == "fixed":
        pdf_chunks = pdf_fixed_chunks(cleaned_pages)
    else:
        pdf_chunks = pdf_paragraph_chunks(cleaned_pages)
    return csv_chunks + pdf_chunks
