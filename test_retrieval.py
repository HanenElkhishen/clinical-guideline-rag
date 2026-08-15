from src.retrieval import retrieve


question = (
    "What is the recommended blood pressure target?"
)


results = retrieve(
    question,
    top_k=5,
)


print("\n")
print("=" * 70)

print("QUESTION:")
print(question)

print("\nTOP RETRIEVED CHUNKS:")

for i, result in enumerate(
    results,
    start=1
):

    print("\n" + "-" * 70)

    print(f"Rank: {i}")

    print(
        "Document:",
        result.document_name
    )

    print(
        "Organization:",
        result.source_organization
    )

    print(
        "Section:",
        result.section_title
    )

    print(
        "Page:",
        result.page_number
    )

    print(
        "Score:",
        result.final_score
    )

    print(
        "Chunk ID:",
        result.chunk_id
    )

    print("\nTEXT:")
    print(result.text[:1000])