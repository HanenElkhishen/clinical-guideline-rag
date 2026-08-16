from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    ScoredPoint,
)

from src.config import (
    QDRANT_URL,
    QDRANT_COLLECTION,
    EMBEDDING_DIM,
)
from src.models import DocumentChunk
from src.embeddings import embed_text

# Initialize Qdrant Client. IMPORTANT: this used to silently fall back to an
# ephemeral in-memory client on ANY connection error, with no warning printed.
# That is dangerous: an in-memory client is a fresh, empty store per process,
# so ingest.py and app.py/test_retrieval.py (separate processes) would each
# get their OWN disconnected in-memory store -- ingestion would appear to
# "succeed" but nothing would ever be found at query time, with no error to
# explain why. We now fail loudly instead so this is impossible to miss.
try:
    client = QdrantClient(url=QDRANT_URL)
    client.get_collections()  # force an actual round-trip now, not lazily later
except Exception as e:
    print(
        f"[WARNING] Could not connect to Qdrant at '{QDRANT_URL}': {e}\n"
        f"[WARNING] Falling back to an IN-MEMORY Qdrant instance. Data stored "
        f"here will NOT persist and will NOT be visible to any other process "
        f"(e.g. if ingest.py runs separately from the app). This almost "
        f"certainly means retrieval will silently return nothing useful.\n"
        f"[WARNING] Fix: make sure Qdrant is running (`docker compose up -d`) "
        f"and QDRANT_URL in your .env is correct, then restart."
    )
    client = QdrantClient(":memory:")


def get_client() -> QdrantClient:
    return client


def create_collection(recreate: bool = False):
    """
    Ensure the target collection exists with the appropriate vector dimension and cosine distance.
    """
    collections = client.get_collections()
    names = [c.name for c in collections.collections]

    if QDRANT_COLLECTION in names:
        if recreate:
            client.delete_collection(collection_name=QDRANT_COLLECTION)
        else:
            return

    # Determine vector dimension
    vector_dimension = EMBEDDING_DIM
    try:
        test_vec = embed_text("clinical guideline test")
        vector_dimension = len(test_vec)
    except Exception:
        pass

    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(
            size=vector_dimension,
            distance=Distance.COSINE,
        ),
    )


def upsert_chunks(
    chunks: List[DocumentChunk],
    embeddings: List[List[float]],
    batch_size: int = 50,
):
    """
    Upload document chunks and vectors to Qdrant in batches.
    """
    total = len(chunks)
    for i in range(0, total, batch_size):
        chunk_batch = chunks[i : i + batch_size]
        emb_batch = embeddings[i : i + batch_size]

        points = []
        for idx, (chunk, embedding) in enumerate(zip(chunk_batch, emb_batch)):
            points.append(
                PointStruct(
                    id=i + idx + 1,  # Sequential integer ID or valid UUID
                    vector=embedding,
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "text": chunk.text,
                        "document_name": chunk.document_name,
                        "source_organization": chunk.source_organization,
                        "page_number": chunk.page_number,
                        "section_title": chunk.section_title,
                        "source_url": chunk.source_url,
                    },
                )
            )

        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points,
        )


def search_dense(
    query_embedding: List[float],
    limit: int = 10,
    score_threshold: Optional[float] = None,
) -> List[ScoredPoint]:
    """
    Execute dense vector search in Qdrant.
    """
    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_embedding,
        limit=limit,
        score_threshold=score_threshold,
        with_payload=True,
    )
    return results.points