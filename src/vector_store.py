from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)

from src.config import (
    QDRANT_URL,
    QDRANT_COLLECTION,
)

from src.models import DocumentChunk
from src.embeddings import embed_text


client = QdrantClient(
    url=QDRANT_URL
)


def create_collection():

    collections = client.get_collections()

    names = [
        c.name
        for c in collections.collections
    ]

    if QDRANT_COLLECTION in names:
        return

    test_vector = embed_text(
        "clinical guideline"
    )

    vector_dimension = len(
        test_vector
    )

    client.create_collection(
        collection_name=QDRANT_COLLECTION,

        vectors_config=VectorParams(
            size=vector_dimension,

            distance=Distance.COSINE,
        ),
    )


def upsert_chunks(
    chunks: list[DocumentChunk],
    embeddings: list[list[float]],
):

    points = []

    for chunk, embedding in zip(
        chunks,
        embeddings
    ):

        points.append(
            PointStruct(
                id=chunk.chunk_id,

                vector=embedding,

                payload={
                    "chunk_id":
                        chunk.chunk_id,

                    "text":
                        chunk.text,

                    "document_name":
                        chunk.document_name,

                    "source_organization":
                        chunk.source_organization,

                    "page_number":
                        chunk.page_number,

                    "section_title":
                        chunk.section_title,

                    "source_url":
                        chunk.source_url,
                },
            )
        )

    client.upsert(
        collection_name=QDRANT_COLLECTION,

        points=points,
    )


def search_dense(
    query_embedding: list[float],
    limit: int = 8,
):

    results = client.query_points(
        collection_name=QDRANT_COLLECTION,

        query=query_embedding,

        limit=limit,

        with_payload=True,
    )

    return results.points