# Experiment Logs

## Chunking Strategy Comparison

### Experiment 1 - Fixed Chunking (400/80)
- Setup: PDF chunk size 400 words with 80 overlap.
- Observation: Strong recall for long sections, but sometimes includes unrelated sentences.
- Retrieval quality: Good for broad policy summaries.

### Experiment 2 - Paragraph-Aware Chunking (300-500 target)
- Setup: Paragraph/section-aware chunking with light overlap.
- Observation: Better semantic coherence and cleaner evidence chunks.
- Retrieval quality: Better precision on targeted factual questions.

## Retrieval Failure Cases
1. Ambiguous query with no clear source relevance:
   - Vector search returned semantically similar but off-topic budget chunks.
2. Numeric-heavy question:
   - BM25 favored chunks with many numbers but wrong context.

## Implemented Fix
- Added domain-specific weighted scoring:
  - source match bonus for election/budget query type
  - keyword overlap bonus
  - year/numeric matching bonus
- Result: More relevant top-4 context selection and fewer weak chunks.
