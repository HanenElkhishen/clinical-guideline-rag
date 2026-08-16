# Evaluation Report

- **Total questions:** 20
- **Model:** qwen3:4b-instruct | **Embedding:** nomic-embed-text | **Top-K:** 5

## Metrics

| Metric | Value |
|---|---|
| Retrieval Precision@5 | 100.0% (14/14) |
| Status Routing Accuracy | 90.0% (18/20) |
| Citation Grounding Accuracy | 88.2% (15/17) |
| Avg Faithfulness Score | 0.815 |

## Failure Analysis

- **Q15** (out_of_scope): What is the capital of France?
  - expected status `insufficient_evidence`, got `answered`
- **Q16** (out_of_scope): What is the recommended treatment for type 2 diabetes?
  - expected status `insufficient_evidence`, got `answered`