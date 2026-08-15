import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows consoles if supported
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.ingestion import extract_pdf_pages
from src.chunking import create_chunks, save_processed_chunks
from src.embeddings import embed_texts
from src.vector_store import (
    create_collection,
    upsert_chunks,
)


DATA_RAW_DIR = Path("data/raw")
DATA_PROCESSED_DIR = Path("data/processed")


def main():
    print("=" * 70)
    print("Clinical Guideline RAG - Ingestion Pipeline")
    print("=" * 70)

    start_time = time.time()

    # Step 1: Initialize Qdrant Collection (recreate for clean slate)
    print("\n[1/5] Initializing Qdrant vector database collection...")
    create_collection(recreate=True)
    print("[OK] Collection ready.")

    # Step 2: Locate and extract PDFs
    print("\n[2/5] Scanning raw clinical guidelines in data/raw/...")
    pdf_files = list(DATA_RAW_DIR.rglob("*.pdf"))

    if not pdf_files:
        print("[ERROR] No PDF files found in data/raw/. Please check the directory.")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF guideline document(s):")
    for p in pdf_files:
        print(f"  - {p} ({p.stat().st_size / 1024:.1f} KB)")

    all_chunks = []
    total_pages_processed = 0

    print("\n[3/5] Extracting and chunking clinical guidelines...")
    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path.name}")
        pages = extract_pdf_pages(str(pdf_path))
        total_pages_processed += len(pages)
        print(f"  Extracted {len(pages)} clean pages.")

        doc_chunks = []
        for page in pages:
            chunks = create_chunks(page)
            doc_chunks.extend(chunks)

        print(f"  Generated {len(doc_chunks)} token-bounded chunks.")
        all_chunks.extend(doc_chunks)

    print(f"\nTotal pages processed: {total_pages_processed}")
    print(f"Total structured chunks created: {len(all_chunks)}")

    # Step 3: Save processed chunks to structured file
    print(f"\n[4/5] Saving processed dataset to {DATA_PROCESSED_DIR}...")
    saved_path = save_processed_chunks(all_chunks, output_dir=str(DATA_PROCESSED_DIR))
    print(f"[OK] Saved {len(all_chunks)} chunks with metadata to: {saved_path}")

    # Step 4: Generate Embeddings and Upsert to Vector Store
    print("\n[5/5] Generating dense embeddings and indexing in Qdrant...")
    texts = [chunk.text for chunk in all_chunks]

    t0 = time.time()
    embeddings = embed_texts(texts, batch_size=32)
    embed_time = time.time() - t0
    rate = len(embeddings) / max(0.001, embed_time)
    print(f"[OK] Generated {len(embeddings)} embeddings in {embed_time:.2f}s ({rate:.1f} chunks/sec).")

    print("Upserting vectors and metadata to Qdrant...")
    upsert_chunks(all_chunks, embeddings, batch_size=50)
    print("[OK] Vector indexing complete.")

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"Ingestion Finished Successfully in {total_time:.2f} seconds!")
    print(f"Artifacts ready in: {DATA_PROCESSED_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()