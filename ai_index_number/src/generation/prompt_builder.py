from __future__ import annotations

from typing import Literal

PromptVersion = Literal["v1", "v2", "v3"]


def build_context_block(selected_chunks: list[dict], max_chars: int = 6500) -> str:
    blocks: list[str] = []
    total = 0
    for ch in selected_chunks:
        block = (
            f"[{ch['chunk_id']}] source={ch['source']} score={ch['final_score']:.3f}\n"
            f"{ch['text']}\n"
        )
        total += len(block)
        if total > max_chars:
            break
        blocks.append(block)
    return "\n".join(blocks)


def build_prompt(query: str, context_block: str, version: PromptVersion = "v3") -> str:
    if version == "v1":
        return (
            "Answer the user using the context below.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {query}\nAnswer:"
        )
    if version == "v2":
        return (
            "You are a careful assistant.\n"
            "Only use the provided context. If context is insufficient, say so.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {query}\n"
            "Answer with concise factual points."
        )
    return (
        "You are the Academic City RAG assistant for CS4241 exam work.\n"
        "Ground all claims in the context chunks only.\n"
        "If the answer is missing, say exactly: "
        "\"I do not have enough information from the provided documents\".\n"
        "Prefer precise numbers and short factual wording.\n\n"
        f"Retrieved Context Chunks:\n{context_block}\n\n"
        f"User Question: {query}\n\n"
        "Return:\n"
        "1) A concise answer\n"
        "2) Supporting chunk IDs used"
    )
