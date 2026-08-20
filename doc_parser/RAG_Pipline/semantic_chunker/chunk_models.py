from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SemanticChunk:
    chunk_id: str
    chunk_type: str
    title: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    teaching_content: dict[str, Any] = field(default_factory=dict)
    parent_chunk: str | None = None
    child_chunks: list[str] = field(default_factory=list)
    related_chunks: list[str] = field(default_factory=list)
    source_nodes: list[str] = field(default_factory=list)
    token_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "chunk_type": self.chunk_type,
            "title": self.title,
            "content": self.content,
            "teaching_content": self.teaching_content,
            "metadata": self.metadata,
            "parent_chunk": self.parent_chunk,
            "child_chunks": self.child_chunks,
            "related_chunks": self.related_chunks,
            "source_nodes": self.source_nodes,
            "token_count": self.token_count,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SemanticChunk":
        return cls(
            chunk_id=payload["chunk_id"],
            chunk_type=payload["chunk_type"],
            title=payload["title"],
            content=payload.get("content", ""),
            metadata=payload.get("metadata", {}),
            teaching_content=payload.get("teaching_content", {}),
            parent_chunk=payload.get("parent_chunk"),
            child_chunks=payload.get("child_chunks", []),
            related_chunks=payload.get("related_chunks", []),
            source_nodes=payload.get("source_nodes", []),
            token_count=payload.get("token_count", 0),
        )
