from __future__ import annotations

import re
from collections import Counter

from .chunk_models import SemanticChunk
from .token_budget import estimate_token_count

STOPWORDS = {
    "the", "a", "an", "and", "or", "for", "with", "from", "into", "this",
    "that", "these", "those", "chapter", "section", "subsection", "paragraph",
    "figure", "source", "activity", "discussion", "glossary", "summary", "example",
}


def _collect_keywords(text: str, title: str | None = None) -> list[str]:
    combined = f"{title or ''} {text or ''}".strip()
    phrases: list[str] = []

    phrase_matches = re.findall(
        r"\b(?:[A-Z][a-zA-Z]+(?:[-'][A-Z]?[a-zA-Z]+)?)(?:\s+(?:[A-Z][a-zA-Z]+(?:[-'][A-Z]?[a-zA-Z]+)?)){0,3}\b",
        combined,
    )
    phrases.extend(phrase_matches)

    hyphen_matches = re.findall(r"\b[a-zA-Z]+(?:-[a-zA-Z]+)+\b", combined)
    phrases.extend(hyphen_matches)

    normalized = []
    for phrase in phrases:
        cleaned = re.sub(r"\s+", " ", phrase).strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in STOPWORDS:
            continue
        normalized.append(cleaned)

    counts = Counter(word.lower() for phrase in normalized for word in phrase.split())
    top_words = [word for word, _ in counts.most_common(8) if word not in STOPWORDS]
    return [word.title() if word.isalpha() else word for word in top_words]


def enrich_chunk(chunk: SemanticChunk, *, chapter: str | None = None, section: str | None = None, subsection: str | None = None) -> SemanticChunk:
    content = chunk.content or ""
    chunk.metadata.setdefault("chapter", chapter)
    chunk.metadata.setdefault("section", section)
    chunk.metadata.setdefault("subsection", subsection)
    chunk.metadata.setdefault("heading_path", [])
    chunk.metadata.setdefault("contains_figures", False)
    chunk.metadata.setdefault("contains_activity", False)
    chunk.metadata.setdefault("contains_source", False)
    chunk.metadata.setdefault("keywords", _collect_keywords(content, chunk.title))
    chunk.token_count = estimate_token_count(content)

    if chunk.chunk_type == "figure":
        chunk.metadata["contains_figures"] = True
    if chunk.chunk_type == "activity":
        chunk.metadata["contains_activity"] = True
    if chunk.chunk_type == "source":
        chunk.metadata["contains_source"] = True

    return chunk
