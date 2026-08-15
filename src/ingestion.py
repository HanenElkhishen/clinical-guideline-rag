from pathlib import Path
import fitz

from src.models import DocumentChunk


ALLOWED_SOURCES = {
    "who": "World Health Organization",
    "cdc": "Centers for Disease Control and Prevention",
    "nice": "National Institute for Health and Care Excellence",
    "uspstf": "U.S. Preventive Services Task Force",
}


def clean_text(text: str) -> str:
    """
    Basic PDF text cleaning.
    """

    text = text.replace("\x00", " ")

    lines = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        lines.append(line)

    return "\n".join(lines)


def extract_pdf_pages(pdf_path: str) -> list[dict]:

    pdf_path = Path(pdf_path)

    source_key = pdf_path.parent.name.lower()

    if source_key not in ALLOWED_SOURCES:
        raise ValueError(
            f"Unsupported source: {source_key}. "
            "Only official guideline sources are allowed."
        )

    source_name = ALLOWED_SOURCES[source_key]

    document = fitz.open(pdf_path)

    pages = []

    for page_index, page in enumerate(document):

        text = page.get_text("text")

        text = clean_text(text)

        if not text:
            continue

        pages.append(
            {
                "document_name": pdf_path.name,
                "source_organization": source_name,
                "page_number": page_index + 1,
                "text": text,
            }
        )

    document.close()

    return pages