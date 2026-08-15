from pydantic import BaseModel, Field
from typing import Literal


class DocumentChunk(BaseModel):
    chunk_id: str
    text: str

    document_name: str
    source_organization: str

    page_number: int
    section_title: str

    source_url: str | None = None


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str

    document_name: str
    source_organization: str

    page_number: int
    section_title: str

    dense_score: float = 0.0
    bm25_score: float = 0.0
    final_score: float = 0.0


class RAGResponse(BaseModel):
    status: Literal[
        "answered",
        "needs_caution",
        "rejected"
    ]

    recommendation: str

    supporting_evidence: list[str]

    citations: list[dict]

    confidence: Literal[
        "HIGH",
        "MEDIUM",
        "LOW"
    ]

    disclaimer: str