from typing import List
import ollama
import time

from src.config import EMBEDDING_MODEL


def embed_text(text: str) -> List[float]:
    """
    Generate embedding for a single text using local Ollama model.
    """
    clean = text.strip()
    if not clean:
        clean = "empty"

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=clean,
    )
    return response["embeddings"][0]


def embed_texts(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """
    Generate embeddings for multiple texts in batches with progress output.
    """
    if not texts:
        return []

    all_embeddings: List[List[float]] = []
    total = len(texts)

    print(f"[EMBED] Starting embeddings for {total} chunks...", flush=True)
    print(f"[EMBED] Model: {EMBEDDING_MODEL}", flush=True)

    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        cleaned_batch = [t.strip() if t.strip() else "empty" for t in batch]

        start_time = time.time()
        batch_end = min(i + len(batch), total)

        print(
            f"[EMBED] Processing {i + 1}-{batch_end}/{total} "
            f"({batch_end / total * 100:.1f}%)...",
            flush=True
        )

        response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=cleaned_batch,
        )

        elapsed = time.time() - start_time

        all_embeddings.extend(response["embeddings"])

        print(
            f"[EMBED] Done {batch_end}/{total} "
            f"in {elapsed:.1f}s",
            flush=True
        )

    print(
        f"[EMBED] Completed: {len(all_embeddings)}/{total} embeddings",
        flush=True
    )

    return all_embeddings