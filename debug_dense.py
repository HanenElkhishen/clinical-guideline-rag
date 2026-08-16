import sys
sys.path.insert(0, ".")
from src.embeddings import embed_text
from src.vector_store import search_dense, get_client
from src.config import QDRANT_COLLECTION

client = get_client()

# 1. Is the client actually talking to real Qdrant, or the silent in-memory fallback?
print("Client type / location:", client._client.__class__.__module__)

# 2. Does the collection exist and how many points does it actually have?
try:
    info = client.get_collection(collection_name=QDRANT_COLLECTION)
    print("Collection exists. Points count:", info.points_count)
    print("Vector size configured:", info.config.params.vectors.size)
except Exception as e:
    print("Could not read collection info:", e)

# 3. Embed the query and check the raw dense search result
vec = embed_text("What is epilepsy?")
print("Query embedding length:", len(vec))

points = search_dense(vec, limit=5)
print("Raw dense search returned", len(points), "points")
for p in points:
    print(" -", getattr(p, "score", None), (p.payload or {}).get("section_title"))