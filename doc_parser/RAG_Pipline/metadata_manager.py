from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ParserContext:
    chapter: Optional[str] = None
    section: Optional[str] = None
    subsection: Optional[str] = None
    page: Optional[int] = None
    inside_chapter: bool = False

    def metadata(self) -> dict:
        return {
            "page": self.page,
            "chapter": self.chapter,
            "section": self.section,
            "subsection": self.subsection,
            "heading_path": self.heading_path(),
        }

    def heading_path(self) -> list[str]:
        path = []
        if self.chapter:
            path.append(self.chapter)
        if self.section:
            path.append(self.section)
        if self.subsection:
            path.append(self.subsection)
        return path

    def update_for_event(self, event: str, title: str) -> None:
        if event == "chapter":
            self.chapter = title
            self.section = None
            self.subsection = None
        elif event == "section":
            self.section = title
            self.subsection = None
        elif event == "subsection":
            self.subsection = title
