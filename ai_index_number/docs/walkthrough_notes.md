# Video Walkthrough Notes

## Recommended Demo Order (8-12 minutes)
1. Introduce project objective and datasets.
2. Explain folder structure and modular code.
3. Show chunking strategy options and why they were chosen.
4. Run Streamlit app and enter sample factual query.
5. Show retrieved chunks, vector/BM25/final scores.
6. Show final prompt and generated answer.
7. Run an adversarial query to show hallucination control.
8. Show evaluation script output JSON.
9. Conclude with innovation feature and lessons learned.

## Key Talking Points
- Manual implementation of all core RAG stages (no LangChain/LlamaIndex).
- Hybrid retrieval with domain-specific weighted scoring.
- Logging and transparency for exam marking.
- RAG vs pure LLM comparison for evidence-based evaluation.
