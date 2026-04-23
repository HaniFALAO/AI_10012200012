from __future__ import annotations

from typing import Any

import numpy as np

from src.retrieval.scoring import (
    classify_query,
    keyword_overlap_bonus,
    normalize_scores,
    source_match_bonus,
    year_numeric_bonus,
)


class HybridRetriever:
    def __init__(self, chunks: list[dict], vector_store: Any, bm25: Any, embedder: Any) -> None:
        self.chunks = chunks
        self.vector_store = vector_store
        self.bm25 = bm25
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 8) -> list[dict]:
        q_type = classify_query(query)
        q_vec = self.embedder.encode([query])[0]

        vec_scores, vec_indices = self.vector_store.search(q_vec, top_k=top_k * 2)
        bm_scores, bm_indices = self.bm25.search(query, top_k=top_k * 2)

        vector_map = {str(idx): float(score) for score, idx in zip(vec_scores, vec_indices) if idx >= 0}
        bm25_map = {str(idx): float(score) for score, idx in zip(bm_scores, bm_indices) if idx >= 0}

        norm_vec = normalize_scores(vector_map)
        norm_bm = normalize_scores(bm25_map)
        all_indices = sorted({*norm_vec.keys(), *norm_bm.keys()})

        results: list[dict] = []
        for idx_str in all_indices:
            idx = int(idx_str)
            chunk = self.chunks[idx]
            v = norm_vec.get(idx_str, 0.0)
            b = norm_bm.get(idx_str, 0.0)
            s_bonus = source_match_bonus(q_type, chunk["source"])
            k_bonus = keyword_overlap_bonus(query, chunk["text"]) * 0.15
            y_bonus = year_numeric_bonus(query, chunk)

            final_score = (0.50 * v) + (0.35 * b) + s_bonus + k_bonus + y_bonus

            results.append(
                {
                    "index": idx,
                    "chunk_id": chunk["chunk_id"],
                    "source": chunk["source"],
                    "text": chunk["text"],
                    "chunk_type": chunk["chunk_type"],
                    "vector_score": float(v),
                    "bm25_score": float(b),
                    "source_bonus": float(s_bonus),
                    "keyword_bonus": float(k_bonus),
                    "year_bonus": float(y_bonus),
                    "final_score": float(final_score),
                    "query_type": q_type,
                }
            )

        ranked = sorted(results, key=lambda x: x["final_score"], reverse=True)
        return ranked[:top_k]


def select_context(ranked_chunks: list[dict], top_n: int = 4, min_score: float = 0.2) -> list[dict]:
    seen: set[str] = set()
    selected: list[dict] = []
    for c in ranked_chunks:
        if c["chunk_id"] in seen:
            continue
        if c["final_score"] < min_score:
            continue
        selected.append(c)
        seen.add(c["chunk_id"])
        if len(selected) >= top_n:
            break
    return selected
