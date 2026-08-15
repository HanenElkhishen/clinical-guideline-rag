from typing import Literal

from pydantic import BaseModel


class QueryClassification(BaseModel):

    category: Literal[
        "allowed",
        "needs_caution",
        "rejected"
    ]

    reason: str


REJECTED_PATTERNS = [
    "diagnose me",
    "diagnose my",
    "what disease do i have",
    "should i take",
    "prescribe",
    "prescription",
]


CAUTION_PATTERNS = [
    "emergency",
    "severe",
    "pregnant",
    "child",
    "infant",
    "dose",
    "dosage",
]


def classify_query(
    question: str
) -> QueryClassification:

    q = question.lower()

    for pattern in REJECTED_PATTERNS:

        if pattern in q:

            return QueryClassification(
                category="rejected",

                reason=(
                    "The question requests "
                    "individualized medical decision-making."
                ),
            )

    for pattern in CAUTION_PATTERNS:

        if pattern in q:

            return QueryClassification(
                category="needs_caution",

                reason=(
                    "The question may involve "
                    "higher-risk clinical context."
                ),
            )

    return QueryClassification(
        category="allowed",

        reason=(
            "Question is suitable for "
            "guideline retrieval."
        ),
    )