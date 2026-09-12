from __future__ import annotations

from collections import OrderedDict
from typing import Any, Mapping
from Backend.retrieval.run_retrieval import retrieve_type
from Backend.retrieval.Inner_LLM import LLM_Input
from Backend.retrieval.outer_LLM import determine_retrieval_method
import json


def router_node(state):
    user_query = state.get("user_query", "").strip()

    if not user_query:
        return {
            "intent": "clarification",
            "Optimised_query": "",
            "Test_Quiz": False,
        }

    try:
        router_data = json.loads(
            determine_retrieval_method(user_query)
        )

        arguments = router_data.get("arguments", {})

        return {
            "intent": router_data.get("trigger", "hybrid_search"),
            "retrieval_method": router_data.get(
                "trigger",
                "hybrid_search"
            ),
            "Optimised_query": (
                arguments.get("Optimised_query")
                or arguments.get("query")
                or user_query
            ),
            "Test_Quiz": arguments.get("Test_Quiz", False),
        }

    except Exception as e:
        print(f"Router error: {type(e).__name__}: {e}")

        return {
            "intent": "hybrid_search",
            "retrieval_method": "hybrid_search",
            "Optimised_query": user_query,
            "Test_Quiz": False,
        }


def retrieval_node(state):
    query = (
        state.get("Optimised_query")
        or state.get("user_query", "")
    ).strip()
    print("QUERY for chunks retrieval:::", query)
    retrieval_method = (
        state.get("retrieval_method")
        or "hybrid_search"
    )
    print("retrieval_method:", retrieval_method)
    if not query:
        return {
            "retrieval_chunks": []
        }

    retrieved_data = retrieve_type(
        query,
        INP=retrieval_method
    )
    
    retrieved_chunks = []

    if isinstance(retrieved_data, list):
        retrieved_chunks = [
            doc.page_content
            if hasattr(doc, "page_content")
            else str(doc)
            for doc in retrieved_data
        ]

    elif isinstance(retrieved_data, dict):
        for section_data in retrieved_data.values():
            retrieved_chunks.extend(
                section_data.get("chunks", [])
            )
    print("retrieved chunks:", retrieved_chunks)
    return {
        "retrieval_chunks": retrieved_chunks
    }
     
def inner_llm_node(state):

    chunks = state.get("retrieval_chunks", [])

    query = (
        state.get("user_query", "")
    )
    print("QUERY for INNER LLM:::", query)
    retrieval_method = (
        state.get("retrieval_method")
        or "hybrid_search"
    )

    if not chunks and retrieval_method not in ("clarification", 
                        "TOC_Overview"
                        # "Student_Progress_Tracking",
                        # "Student_performance_Analysis"
                        ):  #IMP
        return {
            "response": "I could not find enough information to answer your question please ask relevant to the uploaded document, Thank you!"
        }


     
    try:
        response = LLM_Input(
            chunks=chunks,
            query=query,
            top_k=5,
            retrieval_method=retrieval_method,
        )

        return {
            "response": response
        }

    except Exception as e:
        print(f"Inner LLM error: {type(e).__name__}: {e}")

        return {
            "response": "I encountered an error while generating the response."
        }
