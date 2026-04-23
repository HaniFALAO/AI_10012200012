from __future__ import annotations

from pathlib import Path
from typing import Any

from src.generation.prompt_builder import build_context_block, build_prompt
from src.ingestion.load_csv import load_election_csv
from src.ingestion.load_pdf import load_budget_pdf
from src.preprocessing.chunking import build_all_chunks
from src.preprocessing.clean_csv import clean_election_df
from src.preprocessing.clean_pdf import clean_pdf_pages
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.embedder import SentenceEmbedder
from src.retrieval.hybrid_retriever import HybridRetriever, select_context
from src.retrieval.vector_store import FaissVectorStore
from src.utils.helpers import write_json
from src.utils.logger import JsonLogger
from src.utils.paths import project_root, resolve_under_root


class AcademicCityRAG:
    def __init__(
        self,
        csv_path: str = "data/Ghana_Election_Result.csv",
        pdf_path: str = "data/2025_Budget_Statement.pdf",
        chunk_method: str = "paragraph",
        outputs_dir: str = "outputs",
    ) -> None:
        root = project_root()
        self.csv_path = str(resolve_under_root(csv_path))
        self.pdf_path = str(resolve_under_root(pdf_path))
        self.chunk_method = chunk_method
        out = Path(outputs_dir)
        self.outputs_dir = out if out.is_absolute() else (root / out)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.logger = JsonLogger(str(self.outputs_dir / "logs.json"))
        self.ready = False

    def initialize(self) -> None:
        df = clean_election_df(load_election_csv(self.csv_path))
        pages = clean_pdf_pages(load_budget_pdf(self.pdf_path))
        self.chunks = build_all_chunks(df, pages, chunk_method=self.chunk_method)
        write_json(self.outputs_dir / "chunks.json", self.chunks)

        self.embedder = SentenceEmbedder()
        texts = [c["text"] for c in self.chunks]
        vectors = self.embedder.encode(texts)
        self.vector_store = FaissVectorStore(dimension=vectors.shape[1])
        self.vector_store.add(vectors)
        self.bm25 = BM25Retriever(texts)
        self.hybrid = HybridRetriever(self.chunks, self.vector_store, self.bm25, self.embedder)
        self.ready = True

    def answer(
        self,
        query: str,
        llm_client: Any,
        top_k: int = 8,
        prompt_version: str = "v3",
    ) -> dict:
        if not self.ready:
            self.initialize()

        ranked = self.hybrid.retrieve(query, top_k=top_k)
        selected = select_context(ranked, top_n=4, min_score=0.2)
        context = build_context_block(selected)
        prompt = build_prompt(query, context, version=prompt_version)
        answer = llm_client.generate(prompt)
        query_type = ranked[0]["query_type"] if ranked else "mixed"

        result = {
            "query": query,
            "query_type": query_type,
            "retrieved_chunks": ranked,
            "selected_context": selected,
            "final_prompt": prompt,
            "answer": answer,
        }
        self.logger.log_query(result)
        return result
