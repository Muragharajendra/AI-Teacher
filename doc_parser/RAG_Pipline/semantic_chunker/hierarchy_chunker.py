from __future__ import annotations

from ..models import Node
from .chunk_builder import build_chunk_from_node, collect_block_text
from .chunk_rules import classify_chunk_type, create_independent_chunk
from .token_budget import estimate_token_count


class HierarchyChunker:
    def __init__(self) -> None:
        self.chunks: list = []
        self.chunk_lookup: dict[str, str] = {}
        self.node_chunk_map: dict[str, object] = {}
        self.counter = 0

    def _next_id(self) -> str:
        self.counter += 1
        return f"chunk_{self.counter:05d}"

    def build(self, root: Node) -> list:
        self.chunks = []
        self.chunk_lookup = {}
        self.node_chunk_map = {}
        self.counter = 0

        self._walk(root)
        self._link_children(root)
        return self.chunks

    def _walk(self, node: Node, parent_chunk_id: str | None = None) -> None:
        node_type = getattr(node, "node_type", "")
        current_parent = parent_chunk_id

        if node_type in {"chapter", "section", "subsection"}:
            chunk = self._build_hierarchy_chunk(node, parent_chunk_id)
            self.chunks.append(chunk)
            self.chunk_lookup[node.node_id] = chunk.chunk_id
            self.node_chunk_map[node.node_id] = chunk
            current_parent = chunk.chunk_id

        for child in getattr(node, "children", []):
            self._walk(child, current_parent)

        for block in getattr(node, "blocks", []):
            if create_independent_chunk(block.node_type, block.metadata):
                self._build_block_chunk(block, current_parent)

    def _build_hierarchy_chunk(self, node: Node, parent_chunk_id: str | None) -> object:
        content_parts = collect_block_text(node)
        metadata = {
            "chapter": node.metadata.get("chapter"),
            "section": node.metadata.get("section"),
            "subsection": node.metadata.get("subsection"),
            "heading_path": node.metadata.get("heading_path", []),
        }
        if node.node_type == "chapter":
            metadata["is_container"] = True
            metadata["embeddable"] = False

        chunk = build_chunk_from_node(
            node,
            self._next_id(),
            chunk_type=classify_chunk_type(node.node_type),
            title=node.title,
            parent_chunk=parent_chunk_id,
            content_parts=content_parts,
            metadata=metadata,
        )
        chunk.source_nodes = [node.node_id] + [block.node_id for block in getattr(node, "blocks", []) if hasattr(block, "node_id")]
        return chunk

    def _build_block_chunk(self, block: Node, parent_chunk_id: str | None) -> None:
        content = block.content or block.title
        token_count = estimate_token_count(content)
        metadata = {
            "caption": block.metadata.get("caption"),
            "description": block.metadata.get("description"),
            "chapter": block.metadata.get("chapter"),
            "section": block.metadata.get("section"),
            "subsection": block.metadata.get("subsection"),
            "source_length": token_count,
        }

        if block.node_type == "source" and token_count <= 200:
            return

        chunk = build_chunk_from_node(
            block,
            self._next_id(),
            chunk_type=classify_chunk_type(block.node_type),
            title=block.title,
            parent_chunk=parent_chunk_id,
            content_parts=[content],
            metadata=metadata,
        )
        chunk.source_nodes = [block.node_id]
        if parent_chunk_id:
            chunk.related_chunks = [parent_chunk_id]
        self.chunks.append(chunk)

    def _link_children(self, node: Node) -> None:
        if node.node_type in {"chapter", "section", "subsection"}:
            chunk = self.node_chunk_map.get(node.node_id)
            if chunk is not None:
                child_ids = []
                for child in getattr(node, "children", []):
                    child_chunk_id = self.chunk_lookup.get(child.node_id)
                    if child_chunk_id is not None:
                        child_ids.append(child_chunk_id)
                chunk.child_chunks = child_ids

        for child in getattr(node, "children", []):
            self._link_children(child)
