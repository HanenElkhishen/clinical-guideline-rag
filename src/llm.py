import json
import ollama

from src.config import LLM_MODEL
from src.models import RetrievedChunk


SYSTEM_PROMPT = """
You are a clinical guideline retrieval assistant.

Your job is to answer questions ONLY using
the retrieved guideline evidence provided to you.

STRICT RULES:

1. Use ONLY the provided evidence.
2. Do NOT use outside medical knowledge.
3. Do NOT invent recommendations.
4. Do NOT invent citations.
5. Do NOT diagnose individual patients.
6. Do NOT prescribe individualized treatment.
7. If evidence is insufficient, clearly say so.
8. Every citation must correspond to retrieved evidence.
9. Be concise and transparent.
10. Return valid JSON only.

The answer must be grounded in the retrieved
official guideline chunks.
"""


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
):

    evidence_blocks = []

    for index, chunk in enumerate(
        chunks,
        start=1
    ):

        evidence_blocks.append(
            f"""
EVIDENCE #{index}

Document:
{chunk.document_name}

Organization:
{chunk.source_organization}

Section:
{chunk.section_title}

Page:
{chunk.page_number}

Chunk ID:
{chunk.chunk_id}

Text:
{chunk.text}
"""
        )

    evidence = "\n".join(
        evidence_blocks
    )

    user_prompt = f"""
Question:
{question}

Retrieved Evidence:
{evidence}

Return ONLY valid JSON:

{{
    "recommendation": "...",

    "supporting_evidence": [
        "..."
    ],

    "citations": [
        {{
            "document": "...",
            "organization": "...",
            "section": "...",
            "page": 1,
            "chunk_id": "..."
        }}
    ],

    "confidence": "HIGH"
}}
"""

    response = ollama.chat(
        model=LLM_MODEL,

        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],

        options={
            "temperature": 0.1,
        },
    )

    text = response["message"]["content"]

    # Remove markdown fences if model adds them
    text = text.strip()

    if text.startswith("```json"):
        text = text[7:]

    if text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    return json.loads(text)