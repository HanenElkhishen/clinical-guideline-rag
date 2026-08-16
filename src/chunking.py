import json
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
import tiktoken

from src.config import CHUNK_SIZE, CHUNK_OVERLAP
from src.models import DocumentChunk

try:
    encoder = tiktoken.get_encoding("cl100k_base")
except Exception:
    encoder = tiktoken.encoding_for_model("gpt-4o-mini")


def count_tokens(text: str) -> int:
    return len(encoder.encode(text))


def chunk_text_tokens(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping token windows using tiktoken.
    """
    if not text.strip():
        return []

    tokens = encoder.encode(text)
    if len(tokens) <= chunk_size:
        return [text.strip()]

    chunks: List[str] = []
    start = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_decoded = encoder.decode(chunk_tokens).strip()

        if chunk_decoded:
            chunks.append(chunk_decoded)

        if end >= len(tokens):
            break

        start += max(1, chunk_size - chunk_overlap)

    return chunks


def create_chunks(page: Dict[str, Any]) -> List[DocumentChunk]:
    """
    Convert a single processed page dictionary into DocumentChunk instances.
    """
    page_text = page.get("text", "")
    section_title = page.get("section_title", "General Clinical Guidance")
    doc_name = page.get("document_name", "guideline.pdf")
    org_name = page.get("source_organization", "Clinical Authority")
    page_num = page.get("page_number", 1)
    source_url = page.get("source_url", None)

    text_chunks = chunk_text_tokens(page_text)
    result: List[DocumentChunk] = []

    for idx, chunk in enumerate(text_chunks):
        chunk_id = f"{Path(doc_name).stem}_p{page_num}_c{idx+1}_{uuid.uuid4().hex[:6]}"
        tok_cnt = count_tokens(chunk)
        result.append(
            DocumentChunk(
                chunk_id=chunk_id,
                text=chunk,
                document_name=doc_name,
                source_organization=org_name,
                page_number=page_num,
                section_title=section_title,
                source_url=source_url,
                char_count=len(chunk),
                token_count=tok_cnt,
            )
        )

    return result


def save_processed_chunks(chunks: List[DocumentChunk], output_dir: str = "data/processed") -> str:
    """
    Save chunked data into structured JSON and summary files in data/processed/.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    json_file = out_path / "chunks.json"
    chunks_data = [chunk.model_dump() for chunk in chunks]

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)

    # Also save metadata summary
    summary_file = out_path / "dataset_summary.json"
    doc_stats: Dict[str, Any] = {}
    for c in chunks:
        doc = c.document_name
        if doc not in doc_stats:
            doc_stats[doc] = {"total_chunks": 0, "pages": set(), "organization": c.source_organization}
        doc_stats[doc]["total_chunks"] += 1
        doc_stats[doc]["pages"].add(c.page_number)

    formatted_stats = {
        "total_chunks": len(chunks),
        "total_documents": len(doc_stats),
        "documents": {
            k: {
                "organization": v["organization"],
                "total_chunks": v["total_chunks"],
                "total_pages": len(v["pages"]),
            }
            for k, v in doc_stats.items()
        },
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(formatted_stats, f, indent=2)

    # Optional: Save parquet if pandas is available
    try:
        import pandas as pd
        df = pd.DataFrame(chunks_data)
        parquet_file = out_path / "chunks.parquet"
        df.to_parquet(parquet_file, index=False)
    except Exception:
        pass

    return str(json_file)


def load_processed_chunks(processed_file: str = "data/processed/chunks.json") -> List[DocumentChunk]:
    """
    Load structured chunks from data/processed/chunks.json.
    """
    file_path = Path(processed_file)
    if not file_path.exists():
        raise FileNotFoundError(f"Processed chunks file not found at: {processed_file}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [DocumentChunk(**item) for item in data]