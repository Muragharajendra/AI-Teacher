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
# Environment / Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


CHROMA_DIR = BASE_DIR / "docs" / "vectorstore"

COLLECTION_NAME = "ai_teacher"

TOP_K = 5


# ============================================================
# NVIDIA Embedding Model
# ============================================================

class NVIDIAEmbeddings(Embeddings):

    def __init__(self):

        api_key = os.getenv("NVIDIA_API_KEY")

        if not api_key:
            raise ValueError(
                "NVIDIA_API_KEY not found in .env"
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://integrate.api.nvidia.com/v1"
        )

        self.model = "nvidia/nv-embedqa-e5-v5"


    # --------------------------------------------------------
    # Document embeddings
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Query embedding
    # --------------------------------------------------------

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
# Create / Load Persistent Chroma Vector Store
# ============================================================

def create_vectorstore(chunks, rebuild=False):

    embeddings = NVIDIAEmbeddings()


    # --------------------------------------------------------
    # Rebuild existing database
    # --------------------------------------------------------

    if rebuild and CHROMA_DIR.exists():

        print("\nDeleting existing vector database...")

        shutil.rmtree(CHROMA_DIR)

        print("Old vector database deleted.")


    # --------------------------------------------------------
    # Create / Load Chroma
    # --------------------------------------------------------

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


    # --------------------------------------------------------
    # Check existing documents
    # --------------------------------------------------------

    document_count = vectorstore._collection.count()


    # --------------------------------------------------------
    # Create embeddings if database is empty
    # --------------------------------------------------------

    if document_count == 0:

        print("\n========================================")
        print("Creating Vector Database")
        print("========================================")

        print(
            f"Number of chunks: {len(chunks)}"
        )

        print(
            "Generating NVIDIA embeddings..."
        )

        vectorstore.add_documents(chunks)

        print(
            "Embeddings generated and stored."
        )

        print(
            f"Stored documents: "
            f"{vectorstore._collection.count()}"
        )


    # --------------------------------------------------------
    # Reuse existing embeddings
    # --------------------------------------------------------

    else:

        print("\n========================================")
        print("Existing Vector Database Found")
        print("========================================")

        print(
            f"Documents stored: {document_count}"
        )

        print(
            "Skipping embedding generation."
        )


    return vectorstore


# ============================================================
# BM25 Retriever
# ============================================================

def create_bm25_retriever(chunks):

    print("\nCreating BM25 retriever...")

    bm25_retriever = BM25Retriever.from_documents(
        chunks
    )

    bm25_retriever.k = TOP_K

    print(
        f"BM25 top-k: {TOP_K}"
    )

    return bm25_retriever


# ============================================================
# Semantic Retriever
# ============================================================

def create_semantic_retriever(vectorstore):

    print("\nCreating semantic retriever...")

    semantic_retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": TOP_K
        }
    )

    print(
        f"Semantic top-k: {TOP_K}"
    )

    return semantic_retriever


# ============================================================
# Hybrid Retriever
# ============================================================

def create_hybrid_retriever(
    bm25_retriever,
    semantic_retriever
):

    print("\nCreating hybrid retriever...")

    hybrid_retriever = EnsembleRetriever(

        retrievers=[
            bm25_retriever,
            semantic_retriever
        ],

        weights=[
            0.4,    # BM25
            0.6     # Semantic
        ]
    )

    print(
        "\nHybrid weights:"
    )

    print(
        "BM25     : 0.4"
    )

    print(
        "Semantic : 0.6"
    )

    return hybrid_retriever


# ============================================================
# Complete Retrieval System
# ============================================================

def create_retrievers(chunks, rebuild=False):

    # 1. Create / load Chroma
    vectorstore = create_vectorstore(
        chunks,
        rebuild=rebuild
    )

    # 2. Create BM25
    bm25_retriever = create_bm25_retriever(
        chunks
    )

    # 3. Create semantic retriever
    semantic_retriever = create_semantic_retriever(
        vectorstore
    )

    # 4. Create hybrid retriever
    hybrid_retriever = create_hybrid_retriever(
        bm25_retriever,
        semantic_retriever
    )

    # Return all components
    return {
        "vectorstore": vectorstore,
        "bm25": bm25_retriever,
        "semantic": semantic_retriever,
        "hybrid": hybrid_retriever
    }
# ============================================================
# Retrieve Top-K Chunks
# ============================================================

def retrieve_chunks(hybrid_retriever, query, top_k=5):
    """
    Retrieve top-k chunks using hybrid BM25 + semantic retrieval.
    """

    results = hybrid_retriever.invoke(query)

    return results[:top_k]