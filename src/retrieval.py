import re
import json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Optional

from rank_bm25 import BM25Okapi

from src.embeddings import embed_text
from src.vector_store import search_dense
from src.models import RetrievedChunk
from src.config import TOP_K, FINAL_K


# In-memory cache for full-corpus BM25
_CHUNKS_CACHE: Optional[List[Dict[str, Any]]] = None
_BM25_INDEX: Optional[BM25Okapi] = None
_TOKENIZED_CORPUS: Optional[List[List[str]]] = None


def tokenize(text: str) -> List[str]:
    """
    Lowercase and word-tokenize text for BM25.
    """
    return re.findall(r"\b\w+\b", text.lower())


def _load_corpus() -> List[Dict[str, Any]]:
    """
    Load all structured chunks from data/processed/chunks.json.
    """
    global _CHUNKS_CACHE, _BM25_INDEX, _TOKENIZED_CORPUS

    if _CHUNKS_CACHE is not None:
        return _CHUNKS_CACHE

    processed_path = Path("data/processed/chunks.json")
    if not processed_path.exists():
        return []

    with open(processed_path, "r", encoding="utf-8") as f:
        _CHUNKS_CACHE = json.load(f)

    _TOKENIZED_CORPUS = [tokenize(c["text"]) for c in _CHUNKS_CACHE]
    _BM25_INDEX = BM25Okapi(_TOKENIZED_CORPUS)
    return _CHUNKS_CACHE


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """
    Fuse dense and sparse (BM25) search rankings using Reciprocal Rank Fusion.
    Score = sum(1 / (k + rank + 1))
    """
    scores = defaultdict(float)
    payloads: Dict[str, Dict[str, Any]] = {}

    # Weight dense rank
    for rank, item in enumerate(dense_results):
        cid = item["chunk_id"]
        scores[cid] += 1.0 / (k + rank + 1)
        payloads[cid] = item

    # Weight BM25 rank
    for rank, item in enumerate(bm25_results):
        cid = item["chunk_id"]
        scores[cid] += 1.0 / (k + rank + 1)
        if cid not in payloads:
            payloads[cid] = item
        else:
            payloads[cid]["bm25_score"] = item.get("bm25_score", 0.0)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    fused_list: List[Dict[str, Any]] = []
    for cid, rrf_score in ranked:
        item = payloads[cid].copy()
        item["final_score"] = float(rrf_score)
        fused_list.append(item)

    return fused_list


def search_bm25_corpus(query: str, top_k: int = 15) -> List[Dict[str, Any]]:
    """
    Perform BM25 search over the full chunk corpus.
    """
    corpus = _load_corpus()
    if not corpus or _BM25_INDEX is None:
        return []

    q_tokens = tokenize(query)
    if not q_tokens:
        return []

    scores = _BM25_INDEX.get_scores(q_tokens)
    scored_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    results: List[Dict[str, Any]] = []
    for idx in scored_indices[:top_k]:
        score = float(scores[idx])
        if score <= 0.0:
            continue
        chunk = corpus[idx]
        results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "document_name": chunk["document_name"],
                "source_organization": chunk["source_organization"],
                "page_number": chunk["page_number"],
                "section_title": chunk["section_title"],
                "source_url": chunk.get("source_url"),
                "bm25_score": score,
                "dense_score": 0.0,
            }
        )

    return results


def retrieve(
    question: str,
    top_k: int = FINAL_K,
    candidate_k: int = TOP_K * 2,
) -> List[RetrievedChunk]:
    """
    Hybrid retrieval combining Qdrant dense semantic search + full-corpus BM25 with RRF.
    """
    # 1. Dense Semantic Search
    query_embedding = embed_text(question)
    dense_points = search_dense(query_embedding, limit=candidate_k)

    dense_results: List[Dict[str, Any]] = []
    for pt in dense_points:
        payload = pt.payload or {}
        dense_results.append(
            {
                "chunk_id": payload.get("chunk_id", ""),
                "text": payload.get("text", ""),
                "document_name": payload.get("document_name", ""),
                "source_organization": payload.get("source_organization", ""),
                "page_number": payload.get("page_number", 1),
                "section_title": payload.get("section_title", "General"),
                "source_url": payload.get("source_url"),
                "dense_score": float(pt.score) if hasattr(pt, "score") and pt.score else 0.0,
                "bm25_score": 0.0,
            }
        )

    # 2. BM25 Sparse Search across entire corpus
    bm25_results = search_bm25_corpus(question, top_k=candidate_k)

    # 3. Reciprocal Rank Fusion
    fused_candidates = reciprocal_rank_fusion(dense_results, bm25_results, k=60)

    # 4. Map to RetrievedChunk models
    results: List[RetrievedChunk] = []
    for item in fused_candidates[:top_k]:
        results.append(
            RetrievedChunk(
                chunk_id=item["chunk_id"],
                text=item["text"],
                document_name=item["document_name"],
                source_organization=item["source_organization"],
                page_number=item["page_number"],
                section_title=item["section_title"],
                dense_score=item.get("dense_score", 0.0),
                bm25_score=item.get("bm25_score", 0.0),
                final_score=item.get("final_score", 0.0),
                source_url=item.get("source_url"),
            )
        )

    return results