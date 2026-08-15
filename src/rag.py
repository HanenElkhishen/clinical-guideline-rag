from src.config import (
    FINAL_K,
    MIN_RETRIEVAL_SCORE,
)

from src.guardrails import (
    classify_query,
)

from src.retrieval import (
    retrieve,
)

from src.llm import (
    generate_answer,
)


def calculate_confidence(
    chunks
):

    if not chunks:
        return "LOW"

    score = chunks[0].final_score

    if score >= 0.08:
        return "HIGH"

    if score >= 0.04:
        return "MEDIUM"

    return "LOW"


def answer_question(
    question: str
):

    classification = classify_query(
        question
    )

    if classification.category == "rejected":

        return {
            "status": "rejected",

            "recommendation": (
                "I cannot provide individualized "
                "medical diagnosis or treatment decisions."
            ),

            "supporting_evidence": [],

            "citations": [],

            "confidence": "LOW",

            "disclaimer": (
                "This system is for guideline "
                "retrieval and educational use."
            ),
        }

    chunks = retrieve(
        question,
        top_k=FINAL_K,
    )

    if not chunks:

        return {
            "status": "needs_caution",

            "recommendation": (
                "No relevant official guideline "
                "evidence was retrieved."
            ),

            "supporting_evidence": [],

            "citations": [],

            "confidence": "LOW",

            "disclaimer": (
                "No sufficient evidence was found."
            ),
        }

    confidence = calculate_confidence(
        chunks
    )

    result = generate_answer(
        question,
        chunks,
    )

    result["status"] = (
        "needs_caution"
        if classification.category == "needs_caution"
        else "answered"
    )

    result["confidence"] = confidence

    result["disclaimer"] = (
        "This assistant retrieves information "
        "from official clinical guidelines. "
        "It is not a substitute for professional "
        "clinical judgment."
    )

    return result