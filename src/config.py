import os
from dotenv import load_dotenv

load_dotenv()


LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "qwen3:4b-instruct"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "nomic-embed-text"
)

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333"
)

QDRANT_COLLECTION = os.getenv(
    "QDRANT_COLLECTION",
    "clinical_guidelines"
)

TOP_K = int(
    os.getenv("TOP_K", "8")
)

FINAL_K = int(
    os.getenv("FINAL_K", "5")
)

CHUNK_SIZE = int(
    os.getenv("CHUNK_SIZE", "600")
)

CHUNK_OVERLAP = int(
    os.getenv("CHUNK_OVERLAP", "100")
)

MIN_RETRIEVAL_SCORE = float(
    os.getenv(
        "MIN_RETRIEVAL_SCORE",
        "0.35"
    )
)

EMBEDDING_DIM = 768