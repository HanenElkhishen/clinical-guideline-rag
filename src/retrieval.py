import re
from collections import defaultdict

from rank_bm25 import BM25Okapi

from src.embeddings import embed_text
from src.vector_store import search_dense
from src.models import RetrievedChunk


def tokenize(text: str) -> list[str]:

    return re.findall(
        r"\b\w+\b",
        text.lower()
    )


def reciprocal_rank_fusion(
    dense_results,
    bm25_results,
    k: int = 60,
):

    scores = defaultdict(float)

    payloads = {}

    for rank, result in enumerate(
        dense_results
    ):

        chunk_id = result["chunk_id"]

        scores[chunk_id] += (
            1 / (k + rank + 1)
        )

        payloads[chunk_id] = result

    for rank, result in enumerate(
        bm25_results
    ):

        chunk_id = result["chunk_id"]

        scores[chunk_id] += (
            1 / (k + rank + 1)
        )

        payloads[chunk_id] = result

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    return [
        payloads[chunk_id]
        | {"final_score": score}
        for chunk_id, score in ranked
    ]


def retrieve(
    question: str,
    top_k: int = 8,
):

    query_embedding = embed_text(
        question
    )

    dense_points = search_dense(
        query_embedding,
        limit=top_k,
    )

    dense_results = []

    corpus = []

    for point in dense_points:

        payload = point.payload

        item = {
            "chunk_id":
                payload["chunk_id"],

            "text":
                payload["text"],

            "document_name":
                payload["document_name"],

            "source_organization":
                payload["source_organization"],

            "page_number":
                payload["page_number"],

            "section_title":
                payload["section_title"],

            "dense_score":
                point.score,
        }

        dense_results.append(item)

        corpus.append(
            payload["text"]
        )

    # For MVP, use the dense candidates
    # as the BM25 candidate pool.
    tokenized_corpus = [
        tokenize(text)
        for text in corpus
    ]

    if not tokenized_corpus:
        return []

    bm25 = BM25Okapi(
        tokenized_corpus
    )

    query_tokens = tokenize(question)

    bm25_scores = bm25.get_scores(
        query_tokens
    )

    bm25_results = []

    for item, score in zip(
        dense_results,
        bm25_scores
    ):

        bm25_results.append(
            item
            | {
                "bm25_score":
                    float(score)
            }
        )

    bm25_results.sort(
        key=lambda x: x["bm25_score"],
        reverse=True,
    )

    fused = reciprocal_rank_fusion(
        dense_results,
        bm25_results,
    )

    return [
        RetrievedChunk(
            **item
        )
        for item in fused[:top_k]
    ]