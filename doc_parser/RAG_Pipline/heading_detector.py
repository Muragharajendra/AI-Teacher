from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"_(.*?)_", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\u00a0", " ").strip()
    return text


def heading_title(line: str) -> str:
    title = re.sub(r"^\s*#+\s*", "", line).strip()
    return normalize_text(title)


def is_numbered_heading(text: str) -> bool:
    return bool(re.match(r"^\d+(\.\d+)*\s+", text))


def classify_line(line: str, chapter_title: str | None = None):
    text = normalize_text(line)
    if not text:
        return "empty", ""

    # Only accept page markers that are explicit footer/page lines
    if re.search(r"footer page number[: ]*(\d+)", text, re.I) or re.fullmatch(r"page\s*\d+", text, re.I) or re.fullmatch(r"p\.?\s*\d+", text, re.I):
        return "page", ""

    heading_match = re.match(r"^\s*(#+)\s+", line)
    if heading_match:
        title = heading_title(line)
        low_title = title.lower()
        if low_title.startswith("contents") or low_title.startswith("foreword") or low_title.startswith("introduction"):
            return "ignore", ""
        if low_title.startswith("source"):
            return "source", title
        if low_title.startswith("activity"):
            return "activity", title
        if low_title.startswith("discuss") or low_title.startswith("discussion"):
            return "discussion", title
        if low_title.startswith("new words") or low_title.startswith("glossary"):
            return "glossary", title
        if "figure" in low_title or "fig." in low_title or "picture" in low_title:
            return "figure", title
        if "table" in low_title:
            return "table", title
        if "box" in low_title:
            return "box", title
        if "timeline" in low_title:
            return "timeline", title
        if re.match(r"^\d+\.\d+(\.\d+)*\s+", low_title):
            return "subsection", title
        if re.match(r"^\d+\s+(?=[A-Za-z])", low_title):
            return "section", title
        if re.fullmatch(r"section\s+[ivxlcdm]+", low_title):
            return "ignore", ""
        if re.fullmatch(r"section\s+[ivxlcdm]+\s*:\s*.*", low_title):
            return "ignore", ""
        level = len(heading_match.group(1))
        if level == 1:
            return "chapter", title
        if level == 2:
            return "section", title
        return "subsection", title

    low = text.lower()
    if low.startswith("contents") or low.startswith("foreword") or low.startswith("introduction"):
        return "ignore", ""
    if re.fullmatch(r"section\s+[ivxlcdm]+", low):
        return "ignore", ""
    if re.fullmatch(r"section\s+[ivxlcdm]+\s*:\s*.*", low):
        return "ignore", ""

    if re.match(r"^\d+\s+(?=[A-Za-z])", text):
        return "section", text

    if re.match(r"^\d+\.\d+(\.\d+)*\s+(?=[A-Za-z])", text):
        return "subsection", text

    if low.startswith("source"):
        return "source", text
    if low.startswith("activity"):
        return "activity", text
    if low.startswith("discuss") or low.startswith("discussion"):
        return "discussion", text
    if low.startswith("new words") or low.startswith("glossary"):
        return "glossary", text
    if "figure" in low or "fig." in low or "picture" in low:
        return "figure", text
    if "table" in low:
        return "table", text
    if "box" in low:
        return "box", text
    if "timeline" in low:
        return "timeline", text

    if "<!--" in line and "start" in line.lower():
        return "block_marker", text

    return "paragraph", text
