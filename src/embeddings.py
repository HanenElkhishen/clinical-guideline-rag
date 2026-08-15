import ollama

from src.config import EMBEDDING_MODEL


def embed_text(
    text: str
) -> list[float]:

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=text,
    )

    return response["embeddings"][0]


def embed_texts(
    texts: list[str]
) -> list[list[float]]:

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    return response["embeddings"]