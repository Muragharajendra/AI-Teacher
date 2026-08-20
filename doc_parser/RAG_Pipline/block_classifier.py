from __future__ import annotations

import re


def classify_block_type(title: str) -> str:
    text = title.lower()
    if "figure" in text or "fig." in text or "picture" in text:
        return "figure"
    if "table" in text:
        return "table"
    if text.startswith("source"):
        return "source"
    if text.startswith("activity"):
        return "activity"
    if text.startswith("discuss") or text.startswith("discussion"):
        return "discussion"
    if text.startswith("new words") or text.startswith("glossary"):
        return "glossary"
    if "box" in text:
        return "box"
    if "timeline" in text:
        return "timeline"
    return "paragraph"


def parse_glossary_entry(line: str):
    if "–" in line:
        term, definition = line.split("–", 1)
        return {"term": term.strip(), "definition": definition.strip()}
    if "-" in line:
        term, definition = line.split("-", 1)
        return {"term": term.strip(), "definition": definition.strip()}
    return None


def clean_text(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()
