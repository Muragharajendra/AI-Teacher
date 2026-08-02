from __future__ import annotations

import re


def estimate_token_count(text: str) -> int:
    if not text:
        return 0
    token_like = re.findall(r"\b\w+\b|[^\s]", text)
    return max(1, len(token_like))


def budget_bucket(token_count: int) -> str:
    if token_count < 100:
        return "short"
    if token_count < 250:
        return "medium"
    if token_count < 500:
        return "long"
    return "very_long"
