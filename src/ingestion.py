from pathlib import Path
import re
from typing import List, Dict, Any

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        raise ImportError(
            "PyMuPDF is required for PDF ingestion. Please install it using `pip install pymupdf`."
        )


ALLOWED_SOURCES: Dict[str, str] = {
    "who": "World Health Organization",
    "cdc": "Centers for Disease Control and Prevention",
    "nice": "National Institute for Health and Care Excellence",
    "uspstf": "U.S. Preventive Services Task Force",
}

# Patterns to strip out running boilerplate, headers, footers
BOILERPLATE_PATTERNS = [
    r"^Author Manuscript\s*$",
    r"^Author manuscript\s*$",
    r"^HHS Public Access\s*$",
    r"^Epilepsy Res\.\s+Author manuscript.*$",
    r"^Fernandes et al\.\s*$",
    r"^Page \d+\s*$",
    r"^Page \d+ of \d+\s*$",
    r"^Epilepsies in children, young people and adults \(NG217\)\s*$",
    r"^©\s*NICE \d{4}\.\s*All rights reserved\..*$",
    r"^www\.nice\.org\.uk.*$",
    r"^Epilepsy in the WHO Eastern Mediterranean Region: bridging the gap\s*$",
]

COMPILED_BOILERPLATE = [re.compile(p, re.IGNORECASE) for p in BOILERPLATE_PATTERNS]


def clean_text(text: str) -> str:
    """
    Clean raw PDF extracted text:
    - Normalizes Unicode replacement characters, hyphens, quotes, and ligatures.
    - Removes common repeating running headers, footers, and manuscript labels.
    - Normalizes whitespace while preserving paragraphs.
    """
    if not text:
        return ""

    # Replace null bytes
    text = text.replace("\x00", " ")

    # Normalize common ligature and unicode artifacts
    replacements = {
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\xa0": " ",
        "\ufffd": "'",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    cleaned_lines: List[str] = []
    for line in text.splitlines():
        line_str = line.strip()
        if not line_str:
            continue

        # Check if line matches known repetitive boilerplate
        is_boilerplate = any(pattern.match(line_str) for pattern in COMPILED_BOILERPLATE)
        if is_boilerplate:
            continue

        cleaned_lines.append(line_str)

    cleaned_text = "\n".join(cleaned_lines)
    # Collapse 3+ newlines into 2
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    return cleaned_text.strip()


def is_heading_candidate(line: str) -> bool:
    """
    Detect if a line looks like a guideline section heading.
    """
    line_clean = line.strip()
    if not line_clean or len(line_clean) > 100:
        return False

    # Check numbered sections: e.g. "1 Diagnosis and assessment", "1.1 Referral", "6.2 Lennox-Gastaut", "2.2 Historical background"
    if re.match(r"^(\d+(\.\d+)*\s+[A-Z].*)", line_clean):
        return True

    # Check uppercase or title headers
    if re.match(r"^(Recommendations|Overview|Contents|Methods|Results|Discussion|References|Table \d+|Figure \d+|Box \d+|Background)", line_clean, re.IGNORECASE):
        return True

    if line_clean.isupper() and 4 <= len(line_clean) <= 60:
        return True

    return False


def extract_pdf_pages(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract structured pages from a clinical guideline PDF with section tracking.
    """
    path_obj = Path(pdf_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    source_key = path_obj.parent.name.lower()
    if source_key not in ALLOWED_SOURCES:
        # Fallback: check ancestor FOLDER NAMES only (never the filename itself).
        # Matching on the filename (or raw path string) is unsafe here: a file
        # named e.g. "epilepsy_cdc.pdf" would get silently attributed to CDC
        # even if its actual content is an unrelated document, corrupting the
        # citation trail this app relies on for clinical trustworthiness.
        folder_parts = {p.lower() for p in path_obj.parent.parts}
        matched_key = next((key for key in ALLOWED_SOURCES if key in folder_parts), None)

        if matched_key is None:
            raise ValueError(
                f"Unsupported source directory for: {pdf_path}. "
                f"Place the PDF inside a folder named after its source "
                f"(e.g. data/raw/who/, data/raw/cdc/, data/raw/nice/, data/raw/uspstf/). "
                f"Allowed source keys: {list(ALLOWED_SOURCES.keys())}"
            )
        source_name = ALLOWED_SOURCES[matched_key]
    else:
        source_name = ALLOWED_SOURCES[source_key]

    document = fitz.open(str(path_obj))
    pages: List[Dict[str, Any]] = []

    current_section = "General Overview"

    for page_index, page in enumerate(document):
        raw_text = page.get_text("text")
        cleaned = clean_text(raw_text)

        if not cleaned:
            continue

        # Look for section header on this page
        for line in cleaned.splitlines()[:8]:
            if is_heading_candidate(line):
                current_section = line.strip()
                break

        pages.append(
            {
                "document_name": path_obj.name,
                "source_organization": source_name,
                "page_number": page_index + 1,
                "section_title": current_section,
                "text": cleaned,
            }
        )

    document.close()
    return pages