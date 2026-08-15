from typing import Dict, Any

from src.config import FINAL_K
from src.guardrails import (
    classify_query,
    verify_retrieval_confidence,
    verify_unsupported_claims,
)
from src.retrieval import retrieve
from src.llm import generate_answer
from src.models import RAGResponse


DEFAULT_DISCLAIMER = (
    "This system provides evidence retrieval from official published clinical guidelines "
    "(WHO, NICE, CDC) for informational and research purposes only. It is NOT a substitute "
    "for professional clinical judgment, diagnosis, or personalized patient treatment."
)


def answer_question(question: str) -> Dict[str, Any]:
    """
    Main Clinical Guideline RAG Pipeline:
    1. Guardrail Step 1: Input Query Classification (Allowed / Needs Caution / Refused)
    2. Retrieval: Hybrid Dense Semantic Search + Full-Corpus BM25 + Reciprocal Rank Fusion
    3. Guardrail Step 2: Retrieval Relevance & Confidence Verification
    4. Generation: Strict Context-Grounded LLM Generation
    5. Guardrail Step 3: Sentence-level Faithfulness & Unsupported Claim Verification
    6. Formatted Output Construction
    """
    # 1. Input Guardrail
    classification = classify_query(question)

    if classification.category == "rejected":
        return {
            "status": "rejected",
            "recommendation": (
                f"Inquiry Refused: {classification.reason} "
                "Please consult a licensed medical professional for individual diagnosis or treatment."
            ),
            "supporting_evidence": [],
            "citations": [],
            "confidence": "LOW",
            "disclaimer": DEFAULT_DISCLAIMER,
            "faithfulness_score": 1.0,
            "unsupported_claims": [],
        }

    # 2. Hybrid Retrieval
    chunks = retrieve(question, top_k=FINAL_K)

    # 3. Retrieval Confidence & Relevance Verification
    is_sufficient, confidence_rating = verify_retrieval_confidence(chunks)

    if not is_sufficient or not chunks:
        return {
            "status": "insufficient_evidence",
            "recommendation": (
                "No sufficiently relevant evidence found in the official guideline corpus "
                "(CDC, NICE, WHO) to answer this question accurately."
            ),
            "supporting_evidence": [],
            "citations": [],
            "confidence": "INSUFFICIENT_EVIDENCE",
            "disclaimer": DEFAULT_DISCLAIMER,
            "faithfulness_score": 1.0,
            "unsupported_claims": [],
        }

    # 4. Strictly Grounded LLM Generation
    result = generate_answer(question, chunks)

    # 5. Unsupported Claim & Faithfulness Check
    rec_text = result.get("recommendation", "")
    faithfulness, verifications, unsupported = verify_unsupported_claims(rec_text, chunks)

    # Adjust confidence if faithfulness is sub-optimal
    final_confidence = confidence_rating
    if faithfulness < 0.60:
        final_confidence = "LOW"
    elif faithfulness < 0.85 and final_confidence == "HIGH":
        final_confidence = "MEDIUM"

    status = "needs_caution" if classification.category == "needs_caution" else "answered"

    return {
        "status": status,
        "recommendation": rec_text,
        "supporting_evidence": result.get("supporting_evidence", []),
        "citations": result.get("citations", []),
        "confidence": final_confidence,
        "disclaimer": DEFAULT_DISCLAIMER,
        "faithfulness_score": faithfulness,
        "unsupported_claims": unsupported,
    }