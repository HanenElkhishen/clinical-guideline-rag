"""
Evaluation harness for the Clinical Guideline RAG system.

Computes, against a labeled 20-question dataset:
  1. Retrieval Precision@K   -- does at least one of the top-K retrieved
                                 chunks actually contain the expected content?
  2. Status Accuracy         -- does the pipeline route each question to the
                                 correct outcome (answered / needs_caution /
                                 rejected / insufficient_evidence)?
  3. Citation Grounding Accuracy -- for questions that were answered, does the
                                 cited chunk_id actually contain the expected
                                 content (i.e. is the citation trustworthy,
                                 not just present)?
  4. Average Faithfulness    -- the system's own internal faithfulness_score,
                                 averaged across all answered/needs_caution cases.

Usage:
    python evaluate.py

Outputs:
    eval_results.json   -- full raw results per question
    eval_report.md      -- human-readable summary + failure analysis
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, ".")

from src.retrieval import retrieve
from src.rag import answer_question
from src.config import FINAL_K, QDRANT_COLLECTION, LLM_MODEL, EMBEDDING_MODEL


DATASET_PATH = "eval_dataset.json"
CHUNKS_PATH = "data/processed/chunks.json"
RESULTS_PATH = "eval_results.json"
REPORT_PATH = "eval_report.md"


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_chunk_lookup():
    """chunk_id -> chunk text, for citation-grounding verification."""
    path = Path(CHUNKS_PATH)
    if not path.exists():
        print(f"[WARN] {CHUNKS_PATH} not found -- citation grounding check will be skipped.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return {c["chunk_id"]: c["text"] for c in chunks}


def contains_any_keyword(text: str, keywords) -> bool:
    if not keywords:
        return False
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def run_evaluation():
    dataset = load_dataset()
    chunk_lookup = load_chunk_lookup()

    results = []
    retrieval_hits = 0
    retrieval_checked = 0
    status_matches = 0
    citation_correct = 0
    citation_checked = 0
    faithfulness_scores = []

    print("=" * 70)
    print(f"Running evaluation: {len(dataset)} questions | model={LLM_MODEL} | "
          f"embedding={EMBEDDING_MODEL} | collection={QDRANT_COLLECTION} | top_k={FINAL_K}")
    print("=" * 70)

    for case in dataset:
        qid = case["id"]
        question = case["question"]
        expected_status = case["expected_status"]
        expected_keywords = case.get("expected_keywords", [])

        print(f"\n[{qid}] {question}")
        case_result = {
            "id": qid,
            "category": case["category"],
            "question": question,
            "expected_status": expected_status,
        }

        # --- 1. Retrieval Precision@K (only meaningful when we expect real content) ---
        retrieved_hit = None
        if expected_keywords:
            try:
                chunks = retrieve(question, top_k=FINAL_K)
                retrieved_hit = any(
                    contains_any_keyword(c.text, expected_keywords) for c in chunks
                )
                retrieval_checked += 1
                if retrieved_hit:
                    retrieval_hits += 1
            except Exception as e:
                retrieved_hit = False
                retrieval_checked += 1
                case_result["retrieval_error"] = str(e)
        case_result["retrieval_hit_at_k"] = retrieved_hit

        # --- 2. Full pipeline (status accuracy, citation grounding, faithfulness) ---
        try:
            result = answer_question(question)
        except Exception as e:
            case_result["pipeline_error"] = str(e)
            case_result["actual_status"] = "ERROR"
            case_result["status_match"] = False
            results.append(case_result)
            print(f"   [ERROR] pipeline raised: {e}")
            continue

        actual_status = result.get("status")
        case_result["actual_status"] = actual_status
        status_match = actual_status == expected_status
        case_result["status_match"] = status_match
        if status_match:
            status_matches += 1

        if actual_status in ("answered", "needs_caution"):
            faithfulness_scores.append(result.get("faithfulness_score", None))
            case_result["faithfulness_score"] = result.get("faithfulness_score")

        # Citation grounding: only checked when we actually expected groundable content
        if expected_status == "answered" and expected_keywords and actual_status == "answered":
            citations = result.get("citations", [])
            case_result["citations_checked"] = len(citations)
            if citations:
                for cit in citations:
                    citation_checked += 1
                    chunk_text = chunk_lookup.get(cit.get("chunk_id", ""), "")
                    grounded = contains_any_keyword(chunk_text, expected_keywords)
                    if grounded:
                        citation_correct += 1
                case_result["citations_grounded"] = all(
                    contains_any_keyword(chunk_lookup.get(c.get("chunk_id", ""), ""), expected_keywords)
                    for c in citations
                )
            else:
                case_result["citations_grounded"] = False

        print(f"   expected={expected_status:20s} actual={actual_status:20s} "
              f"match={'OK' if status_match else 'FAIL'} "
              f"retrieval_hit={retrieved_hit}")

        results.append(case_result)
        time.sleep(0.2)  # small pause to be gentle on local Ollama

    # --- Aggregate metrics ---
    precision_at_k = retrieval_hits / retrieval_checked if retrieval_checked else None
    status_accuracy = status_matches / len(dataset) if dataset else None
    citation_accuracy = citation_correct / citation_checked if citation_checked else None
    valid_faithfulness = [s for s in faithfulness_scores if isinstance(s, (int, float))]
    avg_faithfulness = sum(valid_faithfulness) / len(valid_faithfulness) if valid_faithfulness else None

    summary = {
        "total_questions": len(dataset),
        "retrieval_precision_at_k": precision_at_k,
        "retrieval_checked_n": retrieval_checked,
        "status_accuracy": status_accuracy,
        "citation_accuracy": citation_accuracy,
        "citation_checked_n": citation_checked,
        "avg_faithfulness": avg_faithfulness,
        "top_k": FINAL_K,
        "model": LLM_MODEL,
        "embedding_model": EMBEDDING_MODEL,
    }

    failures = [r for r in results if not r.get("status_match", False) or r.get("retrieval_hit_at_k") is False]

    # --- Save raw results ---
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)

    # --- Print + save markdown report ---
    report_lines = []
    report_lines.append("# Evaluation Report\n")
    report_lines.append(f"- **Total questions:** {summary['total_questions']}")
    report_lines.append(f"- **Model:** {LLM_MODEL} | **Embedding:** {EMBEDDING_MODEL} | **Top-K:** {FINAL_K}\n")
    report_lines.append("## Metrics\n")
    report_lines.append(f"| Metric | Value |")
    report_lines.append(f"|---|---|")
    report_lines.append(f"| Retrieval Precision@{FINAL_K} | "
                         f"{precision_at_k:.1%} ({retrieval_hits}/{retrieval_checked}) |" if precision_at_k is not None else "| Retrieval Precision@K | N/A |")
    report_lines.append(f"| Status Routing Accuracy | "
                         f"{status_accuracy:.1%} ({status_matches}/{len(dataset)}) |" if status_accuracy is not None else "| Status Routing Accuracy | N/A |")
    report_lines.append(f"| Citation Grounding Accuracy | "
                         f"{citation_accuracy:.1%} ({citation_correct}/{citation_checked}) |" if citation_accuracy is not None else "| Citation Grounding Accuracy | N/A (no citations checked) |")
    report_lines.append(f"| Avg Faithfulness Score | "
                         f"{avg_faithfulness:.3f} |" if avg_faithfulness is not None else "| Avg Faithfulness Score | N/A |")

    report_lines.append("\n## Failure Analysis\n")
    if not failures:
        report_lines.append("No failures — all questions routed and retrieved as expected.")
    else:
        for f_ in failures:
            reason = []
            if not f_.get("status_match", True):
                reason.append(f"expected status `{f_['expected_status']}`, got `{f_.get('actual_status')}`")
            if f_.get("retrieval_hit_at_k") is False:
                reason.append("expected content not found in top-K retrieved chunks")
            if f_.get("pipeline_error"):
                reason.append(f"pipeline error: {f_['pipeline_error']}")
            report_lines.append(f"- **{f_['id']}** ({f_['category']}): {f_['question']}")
            report_lines.append(f"  - {'; '.join(reason)}")

    report_text = "\n".join(report_lines)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

    print("\n" + "=" * 70)
    print(report_text)
    print("=" * 70)
    print(f"\nSaved: {RESULTS_PATH}, {REPORT_PATH}")


if __name__ == "__main__":
    run_evaluation()