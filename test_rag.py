from src.rag import answer_question


question = "What is the meaning of epilepsy?"


print("\n")
print("=" * 70)
print("QUESTION:")
print(question)

print("\nRunning RAG...")
print("=" * 70)


result = answer_question(question)


print("\n")
print("=" * 70)
print("STATUS:")
print(result["status"])

print("\nRECOMMENDATION:")
print(result["recommendation"])

print("\nCONFIDENCE:")
print(result["confidence"])

print("\nSUPPORTING EVIDENCE:")

for evidence in result["supporting_evidence"]:
    print(f"- {evidence}")

print("\nCITATIONS:")

for citation in result["citations"]:
    print(citation)

print("\nDISCLAIMER:")
print(result["disclaimer"])

print("=" * 70)