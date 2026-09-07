from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from .state import TeacherState

from .nodes import (
    router_node,
    retrieval_node,
    inner_llm_node,
)


def build_graph():

    workflow = StateGraph(TeacherState)

    # ========================================================
    # NODES
    # ========================================================

    workflow.add_node(
        "router",
        router_node,
    )

    workflow.add_node(
        "retrieval",
        retrieval_node,
    )

    workflow.add_node(
        "inner_llm",
        inner_llm_node,
    )

    # ========================================================
    # START → ROUTER
    # ========================================================

    workflow.add_edge(
        START,
        "router",
    )

    # ========================================================
    # ROUTER → RETRIEVAL
    # ========================================================

    workflow.add_edge(
        "router",
        "retrieval",
    )

    # ========================================================
    # RETRIEVAL → INNER LLM
    # ========================================================

    workflow.add_edge(
        "retrieval",
        "inner_llm",
    )

    # ========================================================
    # INNER LLM → END
    # ========================================================

    workflow.add_edge(
        "inner_llm",
        END,
    )

    # ========================================================
    # CHECKPOINTER
    # ========================================================

    checkpointer = InMemorySaver()

    # ========================================================
    # COMPILE
    # ========================================================

    graph = workflow.compile(
        checkpointer=checkpointer,
    )

    return graph


# ============================================================
# BUILD GRAPH ONCE
# ============================================================

graph = build_graph()
