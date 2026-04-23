from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from src.evaluation.adversarial_tests import adversarial_queries, factual_queries
from src.generation.llm_client import OllamaClient
from src.pipeline.rag_pipeline import AcademicCityRAG
from src.utils.helpers import write_json


def simple_metrics(answer: str) -> dict:
    lower = answer.lower()
    hallucination_flag = "i do not have enough information from the provided documents" not in lower and (
        "not sure" in lower or "uncertain" in lower
    )
    return {
        "accuracy_proxy": 0.7 if len(answer) > 30 else 0.4,
        "hallucination_proxy": 0.3 if hallucination_flag else 0.1,
        "consistency_proxy": 0.7,
    }


def run_evaluation(output_path: str = "outputs/evaluation_results.json") -> dict:
    load_dotenv()
    rag = AcademicCityRAG()
    llm = OllamaClient()
    all_queries = factual_queries() + adversarial_queries()
    rows: list[dict] = []

    for item in all_queries:
        q = item["query"]
        rag_res = rag.answer(q, llm_client=llm, top_k=8, prompt_version="v3")
        pure = llm.generate_pure_llm(q)
        m = simple_metrics(rag_res["answer"])
        rows.append(
            {
                "id": item["id"],
                "query": q,
                "rag_answer": rag_res["answer"],
                "pure_llm_answer": pure,
                "retrieval_quality_proxy": round(
                    sum(c["final_score"] for c in rag_res["selected_context"]) / max(1, len(rag_res["selected_context"])),
                    4,
                ),
                **m,
            }
        )

    summary = {
        "total_queries": len(rows),
        "avg_accuracy_proxy": round(sum(r["accuracy_proxy"] for r in rows) / len(rows), 3),
        "avg_hallucination_proxy": round(sum(r["hallucination_proxy"] for r in rows) / len(rows), 3),
        "avg_consistency_proxy": round(sum(r["consistency_proxy"] for r in rows) / len(rows), 3),
        "results": rows,
    }
    write_json(Path(output_path), summary)
    return summary


if __name__ == "__main__":
    result = run_evaluation()
    print(f"Saved evaluation for {result['total_queries']} queries.")
