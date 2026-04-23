from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


class BM25Retriever:
    def __init__(self, corpus_texts: list[str]) -> None:
        self.tokens = [_tokenize(t) for t in corpus_texts]
        self.model = BM25Okapi(self.tokens)

    def search(self, query: str, top_k: int = 8) -> tuple[np.ndarray, np.ndarray]:
        q_tokens = _tokenize(query)
        scores = np.array(self.model.get_scores(q_tokens), dtype=float)
        idxs = np.argsort(scores)[::-1][:top_k]
        return scores[idxs], idxs
