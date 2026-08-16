import re
from typing import List, Tuple, Dict, Any, Optional
from pydantic import BaseModel

from src.models import RetrievedChunk, ClaimVerification, QueryClassification
from src.config import MIN_RETRIEVAL_SCORE


# Patterns for strict rejection (individualized diagnosis, personal prescription, harmful actions)
REJECTED_PATTERNS = [
    r"\bdiagnose\s+me\b",
    r"\bdiagnose\s+my\b",
    r"\bwhat\s+disease\s+do\s+i\s+have\b",
    r"\bam\s+i\s+having\s+a\s+seizure\b",
    r"\bshould\s+i\s+take\b",
    r"\bprescribe\s+me\b",
    r"\bwrite\s+me\s+a\s+prescription\b",
    r"\bhow\s+much\s+(dose|dosage)\s+should\s+i\s+take\b",
    r"\bkill\s+myself\b",
    r"\bsuicide\b",
]

# Patterns for caution (high-risk clinical contexts like pregnancy/valproate, emergency status epilepticus, infants)
CAUTION_PATTERNS = [
    r"\bpregnant\b",
    r"\bpregnancy\b",
    r"\bvalproate\s+in\s+pregnancy\b",
    r"\bteratogenic\b",
    r"\bemergency\b",
    r"\bstatus\s+epilepticus\b",
    r"\bprolonged\s+seizure\b",
    r"\bneonate\b",
    r"\binfant\b",
    r"\bdose\b",
    r"\bdosage\b",
]

COMPILED_REJECTED = [re.compile(p, re.IGNORECASE) for p in REJECTED_PATTERNS]
COMPILED_CAUTION = [re.compile(p, re.IGNORECASE) for p in CAUTION_PATTERNS]

STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by", "as",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "and", "or", "but", "if", "then", "else", "when",
    "where", "why", "how", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "can", "will", "just", "should", "now", "it", "its",
    "this", "that", "these", "those", "from", "into", "through", "during", "before",
    "after", "above", "below", "up", "down", "out", "off", "over", "under", "again"
}


def classify_query(question: str) -> QueryClassification:
    """
    Classify input query into: 'allowed', 'needs_caution', or 'rejected'.
    """
    q = question.strip().lower()

    for pattern in COMPILED_REJECTED:
        if pattern.search(q):
            return QueryClassification(
                category="rejected",
                reason=(
                    "The question requests individualized medical diagnosis, direct personal "
                    "prescription, or out-of-scope clinical decision making."
                ),
            )

    for pattern in COMPILED_CAUTION:
        if pattern.search(q):
            return QueryClassification(
                category="needs_caution",
                reason=(
                    "The question involves high-risk clinical context (e.g., pregnancy, "
                    "emergency status epilepticus, pediatric/infant care, or specific dosing)."
                ),
            )

    return QueryClassification(
        category="allowed",
        reason="Question is suitable for official guideline retrieval.",
    )


def verify_retrieval_confidence(
    chunks: List[RetrievedChunk],
    min_score: float = MIN_RETRIEVAL_SCORE,
) -> Tuple[bool, str]:
    """
    Check if retrieved evidence meets the minimum confidence threshold.
    Returns (is_sufficient, confidence_level: HIGH | MEDIUM | LOW | INSUFFICIENT_EVIDENCE)
    """
    if not chunks:
        return False, "INSUFFICIENT_EVIDENCE"

    top_score = chunks[0].final_score
    top_dense = chunks[0].dense_score
    top_bm25 = chunks[0].bm25_score

    # Out-of-domain or irrelevance detection
    if top_score < 0.015 and top_bm25 < 1.0 and top_dense < 0.30:
        return False, "INSUFFICIENT_EVIDENCE"

    if top_score >= 0.030 or top_bm25 >= 8.0 or top_dense >= 0.55:
        return True, "HIGH"

    if top_score >= 0.020 or top_bm25 >= 3.0 or top_dense >= 0.40:
        return True, "MEDIUM"

    return True, "LOW"


# Phrases indicating the LLM itself is reporting that it found no answer.
# If the "recommendation" text matches one of these, it must NEVER be scored
# HIGH/MEDIUM confidence just because it happens to share vocabulary (e.g. the
# word "epilepsy") with the retrieved (and possibly irrelevant) chunks.
REFUSAL_PATTERNS = [
    r"does\s+not\s+contain\s+(a\s+|an\s+)?(sufficient\s+information|definition|explanation|enough\s+information)",
    r"no\s+sufficiently\s+relevant\s+evidence",
    r"insufficient\s+information",
    r"does\s+not\s+contain\s+information",
    r"cannot\s+(be\s+)?(determine|answer|address)",
    r"unable\s+to\s+(find|locate|determine)",
]

COMPILED_REFUSAL = [re.compile(p, re.IGNORECASE) for p in REFUSAL_PATTERNS]


def is_refusal_response(recommendation: str) -> bool:
    """
    Detect whether the generated recommendation is itself a 'no answer found'
    statement. Word-overlap-based faithfulness checks can be fooled by these
    (the refusal text often shares vocabulary, e.g. the disease name, with the
    retrieved context), so this must be checked independently and take
    precedence over the overlap-based faithfulness score.
    """
    if not recommendation:
        return False
    text = recommendation.strip()
    return any(pattern.search(text) for pattern in COMPILED_REFUSAL)


def extract_content_words(text: str) -> set:
    """
    Extract alphanumeric content words excluding common stopwords.
    """
    tokens = re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", text.lower())
    return {t for t in tokens if t not in STOPWORDS}


def verify_unsupported_claims(
    recommendation: str,
    retrieved_chunks: List[RetrievedChunk],
    min_overlap_threshold: float = 0.40,
) -> Tuple[float, List[ClaimVerification], List[str]]:
    """
    Verify each sentence/claim in the generated recommendation against retrieved context.
    Returns:
      - faithfulness_score (0.0 to 1.0)
      - claim_verifications (list of ClaimVerification objects)
      - unsupported_claims (list of sentence strings flagged as unsupported)
    """
    if not recommendation or not retrieved_chunks:
        return 1.0, [], []

    # Concatenate all retrieved text into a single reference pool
    context_text = " ".join([c.text for c in retrieved_chunks]).lower()
    context_words = extract_content_words(context_text)

    # Split recommendation into individual sentences/claims
    raw_sentences = re.split(r"(?<=[.!?])\s+", recommendation.strip())
    sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 10]

    if not sentences:
        return 1.0, [], []

    verifications: List[ClaimVerification] = []
    unsupported: List[str] = []

    for sentence in sentences:
        sent_words = extract_content_words(sentence)
        if not sent_words:
            continue

        # Calculate word overlap with retrieved context
        matched_words = sent_words.intersection(context_words)
        overlap_score = len(matched_words) / len(sent_words)

        # Check if matched chunk can be identified
        best_chunk_id = None
        best_chunk_overlap = 0.0
        for chunk in retrieved_chunks:
            c_words = extract_content_words(chunk.text)
            c_matched = len(sent_words.intersection(c_words))
            if c_matched > best_chunk_overlap:
                best_chunk_overlap = c_matched
                best_chunk_id = chunk.chunk_id

        is_supported = overlap_score >= min_overlap_threshold

        verifications.append(
            ClaimVerification(
                sentence=sentence,
                is_supported=is_supported,
                overlap_score=round(overlap_score, 3),
                matched_chunk_id=best_chunk_id,
            )
        )

        if not is_supported:
            unsupported.append(sentence)

    supported_count = sum(1 for v in verifications if v.is_supported)
    faithfulness_score = round(supported_count / len(verifications), 3) if verifications else 1.0

    return faithfulness_score, verifications, unsupported