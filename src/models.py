from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict, Any


class QueryClassification(BaseModel):
    category: Literal[
        "allowed",
        "needs_caution",
        "rejected"
    ]
    reason: str


class CitationItem(BaseModel):
    document: str
    organization: str
    section: str
    page: int
    chunk_id: str


class DocumentChunk(BaseModel):
    chunk_id: str
    text: str
    document_name: str
    source_organization: str
    page_number: int
    section_title: str
    source_url: Optional[str] = None
    char_count: Optional[int] = None
    token_count: Optional[int] = None


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
    source_url: Optional[str] = None


class ClaimVerification(BaseModel):
    sentence: str
    is_supported: bool
    overlap_score: float = 0.0
    matched_chunk_id: Optional[str] = None


class RAGResponse(BaseModel):
    status: Literal[
        "answered",
        "needs_caution",
        "rejected",
        "insufficient_evidence"
    ]
    recommendation: str
    supporting_evidence: List[str] = Field(default_factory=list)
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: Literal[
        "HIGH",
        "MEDIUM",
        "LOW",
        "INSUFFICIENT_EVIDENCE"
    ]
    disclaimer: str
    faithfulness_score: float = 1.0
    unsupported_claims: List[str] = Field(default_factory=list)