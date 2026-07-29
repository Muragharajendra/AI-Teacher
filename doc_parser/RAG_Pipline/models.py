from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Node:
    node_id: str
    node_type: str
    level: int
    title: str
    content: str = ""
    metadata: dict = field(default_factory=dict)
    parent: Optional["Node"] = None
    children: list["Node"] = field(default_factory=list)
    blocks: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "level": self.level,
            "title": self.title,
            "content": self.content,
            "metadata": self.metadata,
            "children": [child.to_dict() for child in self.children],
            "blocks": [b.to_dict() if hasattr(b, "to_dict") else b for b in self.blocks],
        }
