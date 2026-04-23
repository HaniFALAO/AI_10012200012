# Architecture Notes - Academic City Custom RAG

**Student Name:** `<YOUR_NAME_HERE>`  
**Index Number:** `<YOUR_INDEX_NUMBER_HERE>`

## System Components
1. **Ingestion**
   - `load_csv.py`: reads election CSV with pandas.
   - `load_pdf.py`: extracts text from budget PDF pages using PyMuPDF.

2. **Preprocessing**
   - `clean_csv.py`: normalizes headers, removes duplicates, trims fields.
   - `clean_pdf.py`: normalizes PDF text and removes simple page artifacts.
   - `chunking.py`: creates record chunks for CSV and configurable chunks for PDF.

3. **Embeddings + Vector Storage**
   - `embedder.py`: sentence-transformers embeddings (`all-MiniLM-L6-v2`).
   - `vector_store.py`: FAISS inner-product index for semantic top-k retrieval.

4. **Keyword Retrieval**
   - `bm25_retriever.py`: BM25 scoring over chunk tokens.

5. **Hybrid Retrieval + Domain Scoring**
   - `hybrid_retriever.py`: merges FAISS and BM25 candidates.
   - `scoring.py`: query classifier + weighted score bonuses.
   - Final score uses vector, BM25, source bonus, keyword overlap bonus, and year/numeric bonus.

6. **Prompt Builder + LLM Generation**
   - `prompt_builder.py`: prompt versions v1, v2, v3.
   - `llm_client.py`: local Ollama model generation.

7. **RAG Pipeline**
   - `rag_pipeline.py` orchestrates:
     `Query -> Retrieval -> Context Selection -> Prompt -> LLM -> Response`

8. **UI + Evaluation**
   - `app.py` shows chunks, scores, final prompt, and answer.
   - `run_evaluation.py` and `adversarial_tests.py` support critical testing.

9. **Logging**
   - `logger.py` writes query-level logs to `outputs/logs.json`.

## Data Flow
1. Load and clean CSV/PDF.
2. Chunk and annotate metadata.
3. Embed chunks and build FAISS + BM25 retrievers.
4. Classify query domain.
5. Retrieve and rank chunks with hybrid scoring.
6. Select top context chunks.
7. Build strict grounded prompt.
8. Generate answer via Ollama.
9. Save full trace in JSON logs.

## Architecture Diagram Placeholder
Insert your architecture diagram image here for report/video:
- `docs/architecture_diagram.png` (recommended)
