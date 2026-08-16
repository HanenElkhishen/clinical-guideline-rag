import json
import re
from typing import List, Dict, Any, Optional
import ollama

from src.config import LLM_MODEL, FINAL_K
from src.models import RetrievedChunk


SYSTEM_PROMPT = """You are an expert Clinical Guideline Retrieval Assistant.
Answer the question using ONLY the provided evidence.

RULES:
1. Ground every statement strictly in the provided evidence.
2. Do not extrapolate or prescribe individual treatment.
3. If evidence is insufficient, state: "The retrieved guideline evidence does not contain sufficient information to address this inquiry."
4. Include exact citations matching the evidence.
5. Return ONLY a valid JSON object.
"""


def _format_evidence_context(chunks: List[RetrievedChunk], max_chunks: int = FINAL_K, max_chars_per_chunk: int = 600) -> str:
    blocks = []
    for idx, chunk in enumerate(chunks[:max_chunks], start=1):
        clean_text = chunk.text[:max_chars_per_chunk].strip()
        blocks.append(
            f"[EVIDENCE {idx}]\n"
            f"Document: {chunk.document_name}\n"
            f"Organization: {chunk.source_organization}\n"
            f"Section: {chunk.section_title}\n"
            f"Page: {chunk.page_number}\n"
            f"Chunk ID: {chunk.chunk_id}\n"
            f"Excerpt: {clean_text}\n"
        )
    return "\n".join(blocks)


def _extract_json_payload(raw_text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON object safely from LLM output.
    """
    text = raw_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return None


def generate_answer(
    question: str,
    chunks: List[RetrievedChunk],
    model: str = LLM_MODEL,
) -> Dict[str, Any]:
    """
    Generate grounded clinical recommendation, supporting evidence quotes, and citations.
    """
    if not chunks:
        return {
            "recommendation": "The retrieved guideline evidence does not contain sufficient information to address this inquiry.",
            "supporting_evidence": [],
            "citations": [],
            "confidence": "LOW",
        }

    top_chunks = chunks[:FINAL_K]
    evidence_str = _format_evidence_context(top_chunks)

    user_prompt = f"""Question: {question}

Evidence:
{evidence_str}

Respond with valid JSON:
{{
    "recommendation": "Direct summary based strictly on the evidence above.",
    "supporting_evidence": ["Key quote from evidence."],
    "citations": [
        {{
            "document": "{top_chunks[0].document_name}",
            "organization": "{top_chunks[0].source_organization}",
            "section": "{top_chunks[0].section_title}",
            "page": {top_chunks[0].page_number},
            "chunk_id": "{top_chunks[0].chunk_id}"
        }}
    ],
    "confidence": "HIGH"
}}"""

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": 0.05,
                "num_predict": 450,
            },
        )
        raw_content = response["message"]["content"]
        parsed = _extract_json_payload(raw_content)

        if parsed and isinstance(parsed, dict) and "recommendation" in parsed:
            citations = parsed.get("citations", [])
            valid_citations = []
            for c in citations:
                if isinstance(c, dict):
                    valid_citations.append(
                        {
                            "document": c.get("document", top_chunks[0].document_name),
                            "organization": c.get("organization", top_chunks[0].source_organization),
                            "section": c.get("section", top_chunks[0].section_title),
                            "page": c.get("page", top_chunks[0].page_number),
                            "chunk_id": c.get("chunk_id", top_chunks[0].chunk_id),
                        }
                    )
            parsed["citations"] = valid_citations if valid_citations else [
                {
                    "document": top_chunks[0].document_name,
                    "organization": top_chunks[0].source_organization,
                    "section": top_chunks[0].section_title,
                    "page": top_chunks[0].page_number,
                    "chunk_id": top_chunks[0].chunk_id,
                }
            ]
            return parsed

    except Exception as e:
        print(f"[WARN] LLM generation error: {e}")

    # Fallback to grounded summary
    top_chunk = top_chunks[0]
    return {
        "recommendation": f"According to {top_chunk.source_organization} ({top_chunk.document_name}, Section '{top_chunk.section_title}'): {top_chunk.text[:350]}...",
        "supporting_evidence": [top_chunk.text[:250]],
        "citations": [
            {
                "document": top_chunk.document_name,
                "organization": top_chunk.source_organization,
                "section": top_chunk.section_title,
                "page": top_chunk.page_number,
                "chunk_id": top_chunk.chunk_id,
            }
        ],
        "confidence": "MEDIUM",
    }