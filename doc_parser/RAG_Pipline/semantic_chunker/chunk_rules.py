from __future__ import annotations

from collections.abc import Iterable

BLOCK_TO_CHUNK_TYPE = {
    "figure": "figure",
    "source": "source",
    "activity": "activity",
    "discussion": "discussion",
    "glossary": "glossary",
    "table": "example",
    "box": "summary",
    "timeline": "timeline",
}

HIERARCHY_TO_CHUNK_TYPE = {
    "chapter": "container",
    "section": "section",
    "subsection": "subsection",
}


def classify_chunk_type(node_type: str) -> str:
    if node_type in HIERARCHY_TO_CHUNK_TYPE:
        return HIERARCHY_TO_CHUNK_TYPE[node_type]
    return BLOCK_TO_CHUNK_TYPE.get(node_type, node_type)


def is_hierarchy_node(node_type: str) -> bool:
    return node_type in {"chapter", "section", "subsection"}


def merge_into_parent_chunk(node_type: str) -> bool:
    return node_type in {"paragraph", "glossary", "box", "table", "timeline", "activity", "discussion", "source", "figure"}


def create_independent_chunk(node_type: str, metadata: dict | None = None) -> bool:
    metadata = metadata or {}
    if node_type == "figure":
        return bool(metadata.get("description"))
    if node_type == "source":
        return metadata.get("source_length", 0) > 200
    return node_type in {"activity", "discussion"}


def chunk_excerpt(node_title: str, text: str, limit: int = 150) -> str:
    if not text:
        return ""
    excerpt = text.strip().replace("\n", " ")
    if len(excerpt) <= limit:
        return excerpt
    return excerpt[: limit].rstrip() + "..."


def flatten_children(nodes: Iterable) -> list:
    result: list = []
    for node in nodes:
        result.append(node)
        result.extend(flatten_children(getattr(node, "children", [])))
    return result
