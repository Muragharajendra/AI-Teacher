import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever


# ============================================================
# Environment
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

CHROMA_DIR = BASE_DIR / "docs" / "vectorstore"
COLLECTION_NAME = "ai_teacher"

TOP_K = 5


# ============================================================
# NVIDIA Embeddings
# ============================================================

class NVIDIAEmbeddings(Embeddings):

    def __init__(self):

        api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found in .env"
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )

        self.model = "openai/text-embedding-3-small"


    def embed_documents(self, texts):

        if not texts:
            return []

        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            extra_body={
                "input_type": "passage"
            }
        )

        return [
            item.embedding
            for item in response.data
        ]


    def embed_query(self, text):

        response = self.client.embeddings.create(
            model=self.model,
            input=text,
            extra_body={
                "input_type": "query"
            }
        )

        return response.data[0].embedding


# ============================================================
# Create / Load Chroma
# ============================================================

def create_vector_store(chunks=None, rebuild=False):

    embeddings = NVIDIAEmbeddings()


    # --------------------------------------------------------
    # Rebuild
    # --------------------------------------------------------

    if rebuild and CHROMA_DIR.exists():

        print("Deleting existing Chroma database...")

        shutil.rmtree(CHROMA_DIR)

        print("Old database deleted.")


    # --------------------------------------------------------
    # Open persistent Chroma
    # --------------------------------------------------------

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


    # --------------------------------------------------------
    # Check database
    # --------------------------------------------------------

    document_count = vectorstore._collection.count()


    # --------------------------------------------------------
    # Create database only if empty
    # --------------------------------------------------------

    if document_count == 0:

        if not chunks:
            raise ValueError(
                "Chroma is empty but no chunks were provided."
            )

        print("\nCreating Chroma vector database...")
        print(f"Chunks: {len(chunks)}")

        vectorstore.add_documents(chunks)

        print(
            f"Stored documents: "
            f"{vectorstore._collection.count()}"
        )

    else:

        print("\nExisting Chroma database found.")

        print(
            f"Stored documents: {document_count}"
        )

        print("Skipping embedding generation.")


    return vectorstore


# ============================================================
# Load documents from Chroma
# ============================================================

def load_documents_from_chroma(vectorstore):

    data = vectorstore.get(
        include=["documents", "metadatas"]
    )

    documents = data["documents"]
    metadatas = data["metadatas"]

    from langchain_core.documents import Document

    return [
        Document(
            page_content=text,
            metadata=metadata or {}
        )
        for text, metadata in zip(
            documents,
            metadatas
        )
    ]


# ============================================================
# BM25
# ============================================================

def create_bm25_retriever(documents):

    print("Creating BM25 retriever...")

    bm25 = BM25Retriever.from_documents(
        documents
    )

    bm25.k = TOP_K

    return bm25


# ============================================================
# Semantic Retriever
# ============================================================

def create_semantic_retriever(vectorstore):

    print("Creating semantic retriever...")

    return vectorstore.as_retriever(
        search_kwargs={
            "k": TOP_K
        }
    )


# ============================================================
# Hybrid Retriever
# ============================================================

def create_hybrid_retriever(
    bm25_retriever,
    semantic_retriever
):

    print("Creating hybrid retriever...")

    return EnsembleRetriever(
        retrievers=[
            bm25_retriever,
            semantic_retriever
        ],
        weights=[
            0.4,
            0.6
        ]
    )


# ============================================================
# Initialize ALL retrievers ONCE
# ============================================================

def initialize_retrieval_system(
    chunks=None,
    rebuild=False
):

    print("\n========================================")
    print("INITIALIZING RETRIEVAL SYSTEM")
    print("========================================")


    # 1. Load/create Chroma

    vectorstore = create_vector_store(
        chunks=chunks,
        rebuild=rebuild
    )


    # 2. Load chunks from Chroma

    documents = load_documents_from_chroma(
        vectorstore
    )


    # 3. BM25

    bm25 = create_bm25_retriever(
        documents
    )


    # 4. Semantic

    semantic = create_semantic_retriever(
        vectorstore
    )


    # 5. Hybrid

    hybrid = create_hybrid_retriever(
        bm25,
        semantic
    )


    print("\nRetrieval system ready.")


    return {
        "vectorstore": vectorstore,
        "bm25": bm25,
        "semantic": semantic,
        "hybrid": hybrid
    }


# ============================================================
# Retrieve
# ============================================================

def retrieve_chunks(
    hybrid_retriever,
    query,
    top_k=TOP_K
):

    results = hybrid_retriever.invoke(query)

    return results[:top_k]

# Metadata filter
def metadata_filter(vectorstore, query):
    """
    Filters vectorstore data by query and organizes it into a structured dictionary 
    keyed by unique metadata signatures—optimized for batch LLM processing.
    """
    query = query.strip().lower()

    if not query:
        return {}

    # 1. Fetch raw data from vectorstore
    data = vectorstore.get(include=["documents", "metadatas"])
    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])

    results = []

    # 2. Filter records matching the query string
    for document, metadata in zip(documents, metadatas):
        if not metadata:
            continue
            
        matched = any(
            query in str(value).lower()
            for value in metadata.values()
            if value is not None
        )

        if matched:
            results.append({
                "text": document,
                "metadata": metadata
            })

    if not results:
        return {}

    # 3. Sort chunks chronologically by their reading order index
    results.sort(key=lambda x: x["metadata"].get("chunk_index", float("inf")))

    # 4. Group chunks by their structural heading hierarchy
    grouped_output = {}

    for item in results:
        meta = item["metadata"]
        
        # Create a unique immutable key based on available subheadings
        # (Using a tuple prevents text fragments from mixing across different chapters)
        group_key = (
            meta.get("chapter_name"),
            meta.get("subheading1_name") or meta.get("subheading1"),
            meta.get("subheading2_name") or meta.get("subheading2"),
            meta.get("subheading3_name"),
            meta.get("subheading4_name"),
            meta.get("subheading5_name")
        )

        # Convert tuple key to a safe dictionary string identifier
        group_id = " -> ".join([str(k) for k in group_key if k is not None]) or "General_Section"

        if group_id not in grouped_output:
            grouped_output[group_id] = {
                "metadata": meta,
                "chunks": []
            }
            
        grouped_output[group_id]["chunks"].append(item["text"])

    return grouped_output
