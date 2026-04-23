from __future__ import annotations

import faiss
import numpy as np


class FaissVectorStore:
    def __init__(self, dimension: int) -> None:
        self.index = faiss.IndexFlatIP(dimension)

    def add(self, vectors: np.ndarray) -> None:
        self.index.add(vectors)

    def search(self, query_vector: np.ndarray, top_k: int = 8) -> tuple[np.ndarray, np.ndarray]:
        scores, indices = self.index.search(query_vector.reshape(1, -1), top_k)
        return scores[0], indices[0]
