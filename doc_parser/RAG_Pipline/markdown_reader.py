from __future__ import annotations


def iter_markdown_lines(markdown: str):
    for raw_line in markdown.splitlines():
        yield raw_line.rstrip()
