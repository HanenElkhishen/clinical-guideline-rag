from src.retrieval import retrieve

results = retrieve("What is epilepsy?", top_k=5)
for i, r in enumerate(results, 1):
    print(i, "| score:", round(r.final_score, 4), "| dense:", round(r.dense_score,3), "| bm25:", round(r.bm25_score,2))
    print("   section:", r.section_title, "| page:", r.page_number)
    print("   text:", r.text[:150])
    print()