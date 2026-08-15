import re
import uuid
import tiktoken

from src.config import CHUNK_SIZE, CHUNK_OVERLAP
from src.models import DocumentChunk


encoder = tiktoken.get_encoding("cl100k_base")


def detect_section(text: str) -> str:

    lines = text.splitlines()

    for line in lines:

        clean = line.strip()

        if (
            3 < len(clean) < 120
            and (
                clean.isupper()
                or clean.startswith(
                    (
                        "1.",
                        "2.",
                        "3.",
                        "4.",
                        "5.",
                    )
                )
            )
        ):
            return clean

    return "Unknown Section"


def chunk_text(text: str) -> list[str]:

    tokens = encoder.encode(text)

    chunks = []

    start = 0

    while start < len(tokens):

        end = min(
            start + CHUNK_SIZE,
            len(tokens)
        )

        chunk_tokens = tokens[start:end]

        chunk = encoder.decode(chunk_tokens)

        if chunk.strip():
            chunks.append(chunk.strip())

        if end >= len(tokens):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def create_chunks(page: dict) -> list[DocumentChunk]:

    text_chunks = chunk_text(page["text"])

    section_title = detect_section(page["text"])

    result = []

    for chunk in text_chunks:

        result.append(
            DocumentChunk(
                chunk_id=str(uuid.uuid4()),

                text=chunk,

                document_name=page["document_name"],

                source_organization=page[
                    "source_organization"
                ],

                page_number=page["page_number"],

                section_title=section_title,

                source_url=None,
            )
        )

    return result