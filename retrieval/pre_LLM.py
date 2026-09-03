from run_retrieval import retrieve_type
from Inner_LLM import LLM_Input
top_k=5  # Number of top chunks to retrieve in hybrid search

# query = "What is a nation"
# INP = "metadata_filtering"  # Change to: hybrid_search, metadata_filtering, semantic_retrieval, bm25
def ret_chunks(query, INP):
    # Retrieve data based on INP type
    retrieved_data = retrieve_type(query, INP=INP)

    # Extract chunks into a single list (works for both dict and list responses)
    chunks = []

    if isinstance(retrieved_data, list):
        # hybrid_search, semantic_retrieval, bm25 return list of Document objects
        chunks = [
            doc.page_content if hasattr(doc, 'page_content') else str(doc)
            for doc in retrieved_data
        ]
    elif isinstance(retrieved_data, dict):
        # metadata_filtering returns dict organized by sections
        for section_name, section_data in retrieved_data.items():
            section_chunks = section_data.get("chunks", [])
            chunks.extend(section_chunks)
            
    print(f"Retrieval Method: {INP}")
    print(f"Retrieved {len(chunks)} chunks\n")
    # for i, chunk in enumerate(chunks, 1):
    #     print(f"[Chunk {i}]\n{chunk}\n")
    #     print("-" * 60)
    # return chunks

    # Passing retrieved chunks to Inner_LLM for furthur processing
    LLM_Input(chunks, query=query, top_k=top_k)

# ret_chunks(query, INP)







