from __future__ import annotations

from ..models import Node
from .chunk_models import SemanticChunk
from .chunk_rules import classify_chunk_type, create_independent_chunk, merge_into_parent_chunk
from .context_enricher import enrich_chunk
from .token_budget import estimate_token_count
from .utils import normalize_block_text, safe_list


def _iter_blocks_in_order(node: Node) -> list[Node]:
    blocks = list(safe_list(getattr(node, "blocks", [])))
    for child in safe_list(getattr(node, "children", [])):
        blocks.extend(_iter_blocks_in_order(child))
    return blocks


def _block_content(block: Node) -> str:
    block_type = getattr(block, "node_type", "")
    if block_type == "glossary":
        entries = block.metadata.get("entries", [])
        if entries:
            return "\n".join(
                f"{entry.get('term', 'Term')} – {entry.get('definition', '')}" for entry in entries
            )
        return block.content

    if block_type == "figure":
        caption = block.metadata.get("caption") or block.title
        description = block.metadata.get("description") or block.content
        pieces = [piece for piece in [caption, description] if piece]
        return "\n".join(pieces)

    if block_type == "source":
        return block.content or block.title

    if block_type == "activity":
        return f"{block.title}\n{block.content}".strip()

    return block.content or block.title


def _build_teaching_content(node: Node) -> dict[str, list[str] | dict[str, str]]:
    teaching_content: dict[str, list[str] | dict[str, str]] = {
        "main_text": [],
        "figures": [],
        "sources": [],
        "activities": [],
        "glossary": [],
    }

    for block in _iter_blocks_in_order(node):
        block_type = getattr(block, "node_type", "")
        if block_type == "paragraph":
            text = normalize_block_text(block.content or block.title)
            if text:
                teaching_content["main_text"].append(text)
            continue

        if block_type == "figure":
            caption = normalize_block_text(block.metadata.get("caption") or block.title)
            description = normalize_block_text(block.metadata.get("description") or block.content)
            if caption or description:
                teaching_content["figures"].append({
                    "caption": caption,
                    "description": description,
                })
            continue

        if block_type == "source":
            text = normalize_block_text(block.content or block.title)
            if text:
                teaching_content["sources"].append(text)
            continue

        if block_type == "activity":
            text = normalize_block_text(f"{block.title}\n{block.content}")
            if text:
                teaching_content["activities"].append(text)
            continue

        if block_type == "glossary":
            text = _block_content(block)
            if text:
                teaching_content["glossary"].append(text)

    return teaching_content


def build_chunk_from_node(
    node: Node,
    chunk_id: str,
    *,
    chunk_type: str | None = None,
    title: str | None = None,
    parent_chunk: str | None = None,
    content_parts: list[str] | None = None,
    metadata: dict | None = None,
) -> SemanticChunk:
    chunk_type = chunk_type or classify_chunk_type(node.node_type)
    title = title or node.title
    content_parts = content_parts or []

    is_container = chunk_type == "container" or node.node_type == "chapter"
    flat_content = "" if is_container else "\n\n".join(
        normalize_block_text(part) for part in content_parts if normalize_block_text(part)
    )

    teaching_content = _build_teaching_content(node) if not is_container else {
        "main_text": [],
        "figures": [],
        "sources": [],
        "activities": [],
        "glossary": [],
    }
    chunk = SemanticChunk(
        chunk_id=chunk_id,
        chunk_type=chunk_type,
        title=title,
        content=flat_content,
        teaching_content=teaching_content,
        metadata=metadata or {},
        parent_chunk=parent_chunk,
        source_nodes=[getattr(node, "node_id", "")],
    )
    chunk.token_count = estimate_token_count(flat_content)
    chunk.metadata.setdefault("heading_path", node.metadata.get("heading_path", []))
    chunk.metadata.setdefault("contains_figures", any(teaching_content["figures"]))
    chunk.metadata.setdefault("contains_activity", any(teaching_content["activities"]))
    chunk.metadata.setdefault("contains_source", any(teaching_content["sources"]))
    return enrich_chunk(chunk, chapter=node.metadata.get("chapter"), section=node.metadata.get("section"), subsection=node.metadata.get("subsection"))


def collect_block_text(node: Node) -> list[str]:
    parts: list[str] = []
    for block in _iter_blocks_in_order(node):
        block_type = getattr(block, "node_type", "")
        if not merge_into_parent_chunk(block_type):
            continue
        if block_type == "figure" and not block.metadata.get("description"):
            continue
        text = _block_content(block)
        if text:
            parts.append(text)
    return parts
