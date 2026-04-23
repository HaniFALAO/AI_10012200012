from __future__ import annotations

import os

import ollama


class OllamaClient:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")

    def generate(self, prompt: str, temperature: float = 0.1) -> str:
        resp = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature},
        )
        return resp["message"]["content"].strip()

    def generate_pure_llm(self, query: str, temperature: float = 0.1) -> str:
        prompt = (
            "Answer the user's question directly and concisely.\n"
            "If uncertain, clearly state uncertainty.\n\n"
            f"Question: {query}"
        )
        return self.generate(prompt, temperature=temperature)
