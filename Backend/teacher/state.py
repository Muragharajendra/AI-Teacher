from __future__ import annotations

from typing import Annotated, Any, Optional
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages


class TeacherState(TypedDict, total=False):
    # --------------------------------------------------------
    # USER INPUT
    # --------------------------------------------------------
    user_query: str

    # --------------------------------------------------------
    # CONVERSATION
    # --------------------------------------------------------
    messages: Annotated[list[Any], add_messages]
    history: list[dict]              # [{"user": ..., "tutor": ...}, ...]  (voice)

    # --------------------------------------------------------
    # MODE
    # --------------------------------------------------------
    mode: str                        # "chat" | "voice"

    # --------------------------------------------------------
    # CHAT MODE
    # --------------------------------------------------------
    retrieval_method: str
    Optimised_query: str
    Test_Quiz: bool
    retrieval_chunks: list[str]
    response: str

    # --------------------------------------------------------
    # VOICE MODE
    # --------------------------------------------------------
    session_id: str
    batches: list[list[str]]         # chapter split into batches
    current_batch_index: int
    current_context: str             # active text currently being taught
    active_mode: Optional[str]       # "metadata_filtering" | "hybrid_search" | None
    action: str                      # "answer" | "retrieve"
    intent: str                      # "CONTINUE" | "REPEAT" | "NEW_TOPIC"
    next_node: str                   # internal routing key used by conditional edges      