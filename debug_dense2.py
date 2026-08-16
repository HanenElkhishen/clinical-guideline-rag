import sys
sys.path.insert(0, ".")
from src.embeddings import embed_text
from src.vector_store import search_dense

vec = embed_text("What is epilepsy?")

# Look much deeper than top-5 to see if the definition chunk is buried lower down
points = search_dense(vec, limit=30)
print("Returned", len(points), "points\n")
for i, p in enumerate(points, 1):
    payload = p.payload or {}
    text = payload.get("text", "")
    print(f"{i:2d} | score={p.score:.4f} | {payload.get('document_name')} p{payload.get('page_number')} | {payload.get('section_title')}")
    print(f"     {text[:140].replace(chr(10),' ')}")
    print()