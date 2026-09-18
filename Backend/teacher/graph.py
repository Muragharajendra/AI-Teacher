from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from .state import TeacherState
from .nodes import (
    # chat
    router_node, retrieval_node, inner_llm_node,
    # voice
    voice_entry_node, voice_intent_node, voice_decide_node,
    voice_retrieve_node, voice_advance_batch_node, voice_answer_context_node,
    voice_direct_node, voice_toc_node, voice_impq_node,
    voice_batch_teaching_node, voice_semantic_node,
    route_from_voice_entry, route_from_voice_intent,
    route_from_voice_decide, route_from_voice_retrieve,
)


# ===========================================================================
# CHAT GRAPH  (linear)
# ===========================================================================
def build_chat_graph():
    wf = StateGraph(TeacherState)
    wf.add_node("router", router_node)
    wf.add_node("retrieval", retrieval_node)
    wf.add_node("inner_llm", inner_llm_node)

    wf.add_edge(START, "router")
    wf.add_edge("router", "retrieval")
    wf.add_edge("retrieval", "inner_llm")
    wf.add_edge("inner_llm", END)

    return wf.compile(checkpointer=InMemorySaver())


# ===========================================================================
# VOICE GRAPH  (branching)
# ===========================================================================
def build_voice_graph():
    wf = StateGraph(TeacherState)

    # --- nodes ---
    wf.add_node("voice_entry", voice_entry_node)
    wf.add_node("voice_intent", voice_intent_node)
    wf.add_node("voice_decide", voice_decide_node)
    wf.add_node("voice_retrieve", voice_retrieve_node)
    wf.add_node("voice_advance_batch", voice_advance_batch_node)
    wf.add_node("voice_answer_context", voice_answer_context_node)
    wf.add_node("voice_direct", voice_direct_node)
    wf.add_node("voice_toc", voice_toc_node)
    wf.add_node("voice_impq", voice_impq_node)
    wf.add_node("voice_batch_teaching", voice_batch_teaching_node)
    wf.add_node("voice_semantic", voice_semantic_node)

    # --- entry ---
    wf.add_edge(START, "voice_entry")

    wf.add_conditional_edges(
        "voice_entry",
        route_from_voice_entry,
        {
            "voice_intent": "voice_intent",
            "voice_decide": "voice_decide",
        },
    )

    wf.add_conditional_edges(
        "voice_intent",
        route_from_voice_intent,
        {
            "voice_advance_batch": "voice_advance_batch",
            "voice_answer_context": "voice_answer_context",
            "voice_decide": "voice_decide",
        },
    )

    wf.add_conditional_edges(
        "voice_decide",
        route_from_voice_decide,
        {
            "voice_direct": "voice_direct",
            "voice_retrieve": "voice_retrieve",
        },
    )

    wf.add_conditional_edges(
        "voice_retrieve",
        route_from_voice_retrieve,
        {
            "voice_toc": "voice_toc",
            "voice_impq": "voice_impq",
            "voice_batch_teaching": "voice_batch_teaching",
            "voice_semantic": "voice_semantic",
        },
    )

    # --- terminals ---
    for n in (
        "voice_advance_batch",
        "voice_answer_context",
        "voice_direct",
        "voice_toc",
        "voice_impq",
        "voice_batch_teaching",
        "voice_semantic",
    ):
        wf.add_edge(n, END)

    return wf.compile(checkpointer=InMemorySaver())


# ===========================================================================
# BUILD ONCE
# ===========================================================================
chat_graph  = build_chat_graph()
voice_graph = build_voice_graph()


def get_graph(mode: str = "chat"):
    """Return the compiled graph for the given mode."""
    return voice_graph if mode == "voice" else chat_graph