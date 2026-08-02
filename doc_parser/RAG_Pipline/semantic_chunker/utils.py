from __future__ import annotations

import re
from typing import Iterable


def normalize_content(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_block_text(text: str) -> str:
    return re.sub(r"\n{2,}", "\n", (text or "").strip())


def collapse_lines(lines: Iterable[str]) -> str:
    cleaned = [normalize_content(line) for line in lines if normalize_content(line)]
    return "\n".join(cleaned).strip()


def collect_source_nodes(node) -> list[str]:
    source_ids: list[str] = []
    if getattr(node, "node_id", None):
        source_ids.append(node.node_id)
    for child in getattr(node, "children", []):
        source_ids.extend(collect_source_nodes(child))
    for block in getattr(node, "blocks", []):
        if hasattr(block, "node_id"):
            source_ids.append(block.node_id)
    return source_ids


def safe_list(value):
    return value if isinstance(value, list) else []
