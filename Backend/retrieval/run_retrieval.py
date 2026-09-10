from Backend.doc_parser.chapter_chunking import (
    create_chunks,
    Markdown_extract
)

from Backend.retrieval.retriever import (
    initialize_retrieval_system,
    retrieve_chunks,
    metadata_filter
)


# ============================================================
# 1. Create chunks
# ============================================================
def chunks_creation():
    chunks = create_chunks(Markdown_extract())

    print(
        f"Total chunks created: {len(chunks)}"
    )
    return chunks

# ============================================================
# 2. Initialize retrieval system ONCE
# ============================================================
def retrievers_gen(rebuild):
    retrievers = initialize_retrieval_system(
        chunks=chunks_creation(),
        rebuild=rebuild          # NOTE: MAKE REBUILD fALSE IF YOU RUN THIS CODE 2ND TIME AFTER VECTORE STORAGE.
    )
    return retrievers


retrievers = None

hybrid_retriever = None
semantic_retriever = None
bm25_retriever = None
vectorstore = None


def initialize_retrievers(rebuild=False):

    global retrievers
    global hybrid_retriever
    global semantic_retriever
    global bm25_retriever
    global vectorstore

    retrievers = retrievers_gen(rebuild=rebuild)

    hybrid_retriever = retrievers["hybrid"]
    semantic_retriever = retrievers["semantic"]
    bm25_retriever = retrievers["bm25"]
    vectorstore = retrievers["vectorstore"]
    print("========== Retrieval system initialized ==========")

# ============================================================
# 4. Retrieval function
# ============================================================

def retrieve_type(
    query,
    INP="hybrid_search"
):
    if retrievers is None:
        raise RuntimeError(
            "Retrieval system is not initialized. "
            "Call initialize_retrievers() first."
        )

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

        m_filt= metadata_filter(
            vectorstore,
            query
        )
        if not m_filt:
            return retrieve_chunks(
                        hybrid_retriever,
                        query,
                        top_k=5
                    )
        else:
            return m_filt

    
    elif INP == "Important_Question_Generation":
        m_filt = metadata_filter(
            vectorstore,
            query
        )
        if not m_filt:
            return retrieve_chunks(
                        hybrid_retriever,
                        query,
                        top_k=5
                    )
        else:
            return m_filt
    
# retrieve_type(
#     "European Union",
#     INP="hybrid_search"
# )