# Evaluation Report

## Evaluation Scope
- Factual queries
- Adversarial queries (ambiguous, misleading, incomplete)
- Comparison: RAG vs pure LLM

## Metrics
- Accuracy (proxy/manual review recommended)
- Hallucination rate (proxy + groundedness checks)
- Response consistency
- Retrieval quality (average selected chunk score)

## Observations
1. RAG responses were generally better grounded in source chunks.
2. Pure LLM produced fluent but occasionally unsupported claims.
3. Adversarial queries exposed uncertainty handling behavior.
4. Structured prompt v3 reduced hallucinations compared to v1.

## Evidence
- JSON outputs:
  - `outputs/logs.json`
  - `outputs/evaluation_results.json`

## Examiner Notes
For final submission, add a short manual accuracy review table based on 8-12 queries and cite chunk IDs used for each answer.
