from doc_parser.chapter_chunking import (
    create_chunks,
    markdown_text
)

from retrieval.retriever import (
    initialize_retrieval_system,
    retrieve_chunks,
    metadata_filter
)


# ============================================================
# 1. Create chunks
# ============================================================

chunks = create_chunks(markdown_text)

print(
    f"Total chunks created: {len(chunks)}"
)

# ============================================================
# 2. Initialize retrieval system ONCE
# ============================================================

retrievers = initialize_retrieval_system(
    chunks=chunks,
    rebuild=False
)


# ============================================================
# 3. Get retrievers
# ============================================================

hybrid_retriever = retrievers["hybrid"]

semantic_retriever = retrievers["semantic"]

bm25_retriever = retrievers["bm25"]

vectorstore = retrievers["vectorstore"]


# ============================================================
# 4. Retrieval function
# ============================================================

def retrieve_type(
    query,
    INP="hybrid_search"
):

    if INP == "semantic_retrieval":

        return semantic_retriever.invoke(query)


    elif INP == "bm25":

        return bm25_retriever.invoke(query)


    elif INP == "hybrid_search":

        return retrieve_chunks(
            hybrid_retriever,
            query,
            top_k=5
        )

    elif INP == "metadata_filtering":

        return metadata_filter(
            vectorstore,
            query
        )

    else:

        raise ValueError(
            f"Unknown retrieval type: {INP}"
        )
# retrieve_type(
#     "European Union",
#     INP="hybrid_search"
# )