# Academic City Custom RAG Chatbot (CS4241 Exam Project)

**Student Name:** `<Hanif_CHITOU>`  
**Index Number:** `<10012200012>`

## Overview
This project implements a full **custom Retrieval-Augmented Generation (RAG)** chatbot for Academic City using:
- Ghana election results dataset (CSV)
- 2025 Ghana budget statement (PDF)

The system is intentionally built **without LangChain/LlamaIndex**. All core RAG components are implemented manually for exam transparency.

## Stack
- Python
- Streamlit
- pandas
- PyMuPDF
- sentence-transformers (`all-MiniLM-L6-v2`)
- FAISS
- NumPy
- scikit-learn
- rank-bm25
- python-dotenv
- Ollama (local LLM)

## Architecture Summary
Pipeline:
`User Query -> Hybrid Retrieval (FAISS + BM25) -> Context Selection -> Prompt Builder -> Ollama LLM -> Response`

Core modules:
- `src/ingestion`: CSV/PDF loading
- `src/preprocessing`: cleaning + chunking
- `src/retrieval`: embeddings, FAISS, BM25, hybrid retrieval, scoring
- `src/generation`: prompt templates + Ollama client
- `src/pipeline`: end-to-end RAG pipeline
- `src/evaluation`: adversarial testing and evaluation
- `src/utils`: helpers + JSON logging

## Setup
1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install Ollama and pull a model:
   ```bash
   ollama pull llama3.1:8b
   ```
4. Copy env file:
   ```bash
   copy .env.example .env
   ```

## Environment Variables
- `OLLAMA_MODEL`: local model name to use (default: `llama3.1:8b`)

## Run Locally
```bash
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. **Main file:** set to `ai_index_number/app.py` (or your equivalent path).
2. **Python requirements:** `ai_index_number/requirements.txt`.
3. **Commit datasets:** `data/Ghana_Election_Result.csv` and `data/2025_Budget_Statement.pdf` must be in the GitHub repo (not only on your laptop). If they are large, use [Git LFS](https://git-lfs.github.com/).
4. **Paths:** the app resolves `data/` and `outputs/` from the `ai_index_number` folder so it works whether the process cwd is the repo root or the app folder.
5. **LLM on Streamlit Cloud:** there is no local Ollama. Either add **`OPENAI_API_KEY`** (and optionally `OPENAI_MODEL`) in **App settings → Secrets** so the app falls back to OpenAI after Ollama fails, or set **`OLLAMA_HOST`** to a public Ollama URL you control. Otherwise the chat will show a clear “LLM unavailable” message instead of crashing.

## How Retrieval Works
1. CSV rows are converted into natural-language record chunks.
2. PDF is chunked using either:
   - fixed-size (400 words, 80 overlap), or
   - paragraph-aware chunking (target 300-500 words with light overlap).
3. Embeddings are generated using `all-MiniLM-L6-v2`.
4. FAISS returns semantic candidates.
5. BM25 returns keyword candidates.
6. Hybrid scoring merges both plus domain bonuses:
   - vector similarity
   - BM25 score
   - source match bonus
   - keyword overlap bonus
   - year/numeric bonus
7. Top context chunks are selected and injected into prompt templates.

## Innovation Feature
Custom domain-specific weighted scoring in `src/retrieval/hybrid_retriever.py`:
- Improves grounding for election-only vs budget-only queries.
- Rewards chunks with numeric/year alignment for factual finance/election questions.

## Evaluation Summary
Evaluation script compares:
- RAG mode (retrieval + grounding)
- Pure LLM mode (no retrieval)

Metrics/proxies recorded:
- accuracy
- hallucination rate
- consistency
- retrieval quality

Results are saved to `outputs/evaluation_results.json`.

## Exam Deliverables Included
- Streamlit app
- Modular source code
- Logging pipeline (`outputs/logs.json`)
- Chunk outputs (`outputs/chunks.json`)
- Evaluation script and adversarial tests
- Architecture and report documents in `docs/`

## Notes
- The code is designed for readability and examiner review.
- Add your name and index number in this README and optionally in file headers/comments before submission.
