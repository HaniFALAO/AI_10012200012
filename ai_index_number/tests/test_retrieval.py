from src.generation.prompt_builder import build_prompt
from src.retrieval.hybrid_retriever import select_context
from src.retrieval.scoring import normalize_scores


def test_normalize_scores() -> None:
    norm = normalize_scores({"a": 2.0, "b": 4.0})
    assert 0.0 <= norm["a"] < norm["b"] <= 1.0


def test_select_context_top_n() -> None:
    ranked = [
        {"chunk_id": "c1", "final_score": 0.7},
        {"chunk_id": "c2", "final_score": 0.6},
        {"chunk_id": "c1", "final_score": 0.5},
    ]
    selected = select_context(ranked, top_n=2, min_score=0.2)
    assert len(selected) == 2
    assert selected[0]["chunk_id"] == "c1"


def test_prompt_creation() -> None:
    prompt = build_prompt("What is revenue?", "[c1] source=budget_pdf score=0.9\nRevenue info", version="v3")
    assert "provided documents" in prompt
    assert "User Question" in prompt
