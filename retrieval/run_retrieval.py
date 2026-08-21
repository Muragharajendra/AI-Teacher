from doc_parser.chapter_chunking import (
    create_chunks,
    markdown_text
)

from retrieval.retriever import (
    create_retrievers,
    retrieve_chunks
)


# ============================================================
# 1. Create chunks
# ============================================================

chunks = create_chunks(markdown_text)

print(
    f"Chunks created: {len(chunks)}"
)


# ============================================================
# 2. Create retrieval system
# ============================================================

retrievers = create_retrievers(
    chunks,
    rebuild=False
)


# ============================================================
# 3. Get hybrid retriever
# ============================================================

hybrid_retriever = retrievers["hybrid"]


# ============================================================
# 4. Query
# ============================================================

query = "What is a Absolutist?"


# ============================================================
# 5. Retrieve top 5 chunks
# ============================================================

retrieved_chunks = retrieve_chunks(
    hybrid_retriever,
    query,
    top_k=5
)

print("Top - k chunks retrieved successfully.")

# ============================================================
# 6. Print results
# ============================================================

print("\n")
print("=" * 80)
print("RETRIEVED CHUNKS")
print("=" * 80)

for i, chunk in enumerate(
    retrieved_chunks,
    start=1
):

    print(f"\n--- CHUNK {i} ---")

    print("Metadata:")
    print(chunk.metadata)

    print("\nContent:")
    print(chunk.page_content)