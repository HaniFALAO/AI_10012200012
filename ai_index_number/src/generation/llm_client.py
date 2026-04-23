from __future__ import annotations

import os

import httpx
import ollama


class OllamaClient:
    """Local Ollama by default; optional OpenAI-compatible HTTP API when Ollama is unreachable."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")

    def _openai_compatible_chat(self, prompt: str, temperature: float) -> str:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("missing_openai_key")
        base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        chat_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        url = f"{base}/chat/completions"
        with httpx.Client(timeout=120.0) as client:
            r = client.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": chat_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                },
            )
            r.raise_for_status()
            data = r.json()
        return str(data["choices"][0]["message"]["content"]).strip()

    def generate(self, prompt: str, temperature: float = 0.1) -> str:
        try:
            resp = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": temperature},
            )
            return resp["message"]["content"].strip()
        except Exception as exc:
            if os.getenv("OPENAI_API_KEY", "").strip():
                try:
                    return self._openai_compatible_chat(prompt, temperature)
                except Exception as fallback_exc:
                    return (
                        "**LLM unavailable**\n\n"
                        "Ollama could not be reached, and the OpenAI fallback also failed.\n\n"
                        f"- Ollama error: `{exc!s}`\n"
                        f"- Fallback error: `{fallback_exc!s}`\n\n"
                        "For **Streamlit Cloud**, set `OPENAI_API_KEY` in app secrets, or expose Ollama "
                        "and set `OLLAMA_HOST` to that URL. For **local** use, run Ollama on this machine."
                    )
            return (
                "**LLM unavailable (Ollama not reachable)**\n\n"
                "Streamlit Cloud has no local Ollama server. Choose one:\n\n"
                "1. **OpenAI (recommended for cloud):** add `OPENAI_API_KEY` (and optionally "
                "`OPENAI_MODEL`, `OPENAI_BASE_URL`) in Streamlit **Secrets** or `.env`.\n\n"
                "2. **Remote Ollama:** run Ollama on a server you control and set `OLLAMA_HOST` "
                "(for example `https://your-host:11434`) in secrets.\n\n"
                "3. **Local demo:** run `streamlit run app.py` on your PC with Ollama installed.\n\n"
                f"Technical detail: `{exc!s}`"
            )

    def generate_pure_llm(self, query: str, temperature: float = 0.1) -> str:
        prompt = (
            "Answer the user's question directly and concisely.\n"
            "If uncertain, clearly state uncertainty.\n\n"
            f"Question: {query}"
        )
        return self.generate(prompt, temperature=temperature)
