
from __future__ import annotations

from typing import Annotated, Any
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages


class TeacherState(TypedDict, total=False):

    # ========================================================
    # USER INPUT
    # ========================================================

    user_query: str

    # ========================================================
    # CONVERSATION
    # ========================================================

    messages: Annotated[list[Any], add_messages]

    # ========================================================
    # LLM ROUTER OUTPUT
    # ========================================================
    retrieval_method: str
    Optimised_query: str
    Test_Quiz: bool

    # ========================================================
    # RETRIEVAL OUTPUT
    # ========================================================

    retrieval_chunks: list[str]

    # ========================================================
    # FINAL RESPONSE
    # ========================================================
    response: str

