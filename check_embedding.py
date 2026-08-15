from src.embeddings import embed_text


vector = embed_text(
    "What is hypertension?"
)

print("Embedding dimension:")
print(len(vector))