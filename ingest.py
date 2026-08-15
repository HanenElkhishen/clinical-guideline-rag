from pathlib import Path

from src.ingestion import extract_pdf_pages
from src.chunking import create_chunks
from src.embeddings import embed_texts
from src.vector_store import (
    create_collection,
    upsert_chunks,
)


DATA_DIR = Path("data/raw")


def main():

    print("=" * 60)
    print("Clinical Guideline RAG - Ingestion")
    print("=" * 60)

    create_collection()

    all_chunks = []

    pdf_files = list(
        DATA_DIR.rglob("*.pdf")
    )

    if not pdf_files:
        print("No PDF files found.")
        return

    print(
        f"Found {len(pdf_files)} PDF files."
    )

    for pdf_path in pdf_files:

        print(
            f"\nProcessing: {pdf_path}"
        )

        pages = extract_pdf_pages(
            str(pdf_path)
        )

        for page in pages:

            chunks = create_chunks(page)

            all_chunks.extend(chunks)

    print(
        f"\nCreated {len(all_chunks)} chunks."
    )

    texts = [
        chunk.text
        for chunk in all_chunks
    ]

    print("Generating embeddings...")

    embeddings = embed_texts(texts)

    print("Uploading to Qdrant...")

    upsert_chunks(
        all_chunks,
        embeddings
    )

    print("\nDone.")


if __name__ == "__main__":
    main()