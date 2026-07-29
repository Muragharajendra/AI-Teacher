from __future__ import annotations

import json
import re
from typing import Optional

from .block_classifier import classify_block_type, clean_text, parse_glossary_entry
from .heading_detector import classify_line, heading_title, normalize_text
from .markdown_reader import iter_markdown_lines
from .metadata_manager import ParserContext
from .models import Node


def make_node_id(node_type: str, counter: dict[str, int]) -> str:
    counter[node_type] = counter.get(node_type, 0) + 1
    return f"{node_type}_{counter[node_type]:05d}"


def create_node(node_type: str, title: str, parent: Node, level: int, metadata: dict, counter: dict[str, int], content: str = "") -> Node:
    return Node(
        node_id=make_node_id(node_type, counter),
        node_type=node_type,
        level=level,
        title=title,
        content=content,
        metadata=metadata,
        parent=parent,
    )


def build_semantic_tree(markdown: str, chapter_title: str = "Untitled Chapter") -> Node:
    root = Node(node_id="root", node_type="root", level=0, title="ROOT")
    context = ParserContext(chapter=chapter_title)
    # After detecting a chapter heading we will ignore front-matter (contents, foreword)
    # until we see an Introduction or the first numbered/section heading.
    context.inside_chapter = False

    current_chapter: Optional[Node] = None
    current_section: Optional[Node] = None
    current_subsection: Optional[Node] = None
    current_container: Optional[Node] = None
    active_block: Optional[Node] = None
    pending_paragraph_lines: list[str] = []
    counter: dict[str, int] = {}

    def flush_paragraph():
        nonlocal pending_paragraph_lines, active_block
        if not pending_paragraph_lines:
            return
        text = "\n".join(clean_text(line) for line in pending_paragraph_lines if clean_text(line))
        pending_paragraph_lines = []
        if not text:
            return

        if active_block is not None and active_block.node_type in {"figure", "source", "glossary", "discussion", "activity", "table", "box", "timeline"}:
            if active_block.node_type == "glossary":
                entries = active_block.metadata.get("entries", [])
                for entry_line in text.splitlines():
                    parsed = parse_glossary_entry(entry_line)
                    if parsed:
                        entries.append(parsed)
                if entries:
                    active_block.metadata["entries"] = entries
                return

            if active_block.node_type == "source":
                body = (active_block.content + "\n" + text).strip()
                active_block.content = body
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                if lines:
                    first = lines[0]
                    if first.lower().startswith("by "):
                        active_block.metadata["author"] = first[3:].strip()
                    elif len(first.split()) <= 5 and re.match(r"^[A-Z][A-Za-z\s,'-]+$", first):
                        active_block.metadata.setdefault("author", first)
                return

            if active_block.node_type == "figure":
                active_block.metadata["description"] = text
                return

            active_block.content = (active_block.content + "\n" + text).strip()
            return

        parent = current_container or current_subsection or current_section or current_chapter or root
        paragraph_node = create_node(
            "paragraph",
            "Paragraph",
            parent,
            parent.level + 1,
            context.metadata(),
            counter,
            content=text,
        )
        parent.blocks.append(paragraph_node)

    def create_block(node_type: str, title: str, content: str = "") -> Node:
        parent = current_container or current_subsection or current_section or current_chapter or root
        block_node = create_node(
            node_type,
            title,
            parent,
            parent.level + 1,
            context.metadata(),
            counter,
            content=content,
        )
        if node_type == "glossary":
            block_node.metadata["entries"] = []
        if node_type == "source":
            block_node.metadata["type"] = "source"
        if node_type == "figure" and title:
            block_node.metadata["caption"] = title
        parent.blocks.append(block_node)
        return block_node

    for line in iter_markdown_lines(markdown):
        normalized = normalize_text(line)
        if not normalized:
            flush_paragraph()
            continue

        event, title = classify_line(line, chapter_title=chapter_title)

        if event == "ignore":
            flush_paragraph()
            active_block = None
            continue

        if event == "page":
            page_match = re.search(r"footer page number[: ]*(\d+)", line, re.I)
            if page_match:
                context.page = int(page_match.group(1))
            active_block = None
            continue

        if event == "chapter":
            flush_paragraph()
            active_block = None
            context.chapter = title or chapter_title
            context.section = None
            context.subsection = None
            current_chapter = create_node(
                "chapter",
                context.chapter,
                root,
                1,
                context.metadata(),
                counter,
            )
            root.children.append(current_chapter)
            current_section = None
            current_subsection = None
            current_container = None
            context.inside_chapter = False
            continue

        if event == "section":
            if current_chapter is None:
                current_chapter = create_node("chapter", chapter_title, root, 1, context.metadata(), counter)
                root.children.append(current_chapter)
            flush_paragraph()
            active_block = None
            context.update_for_event("section", title)
            current_section = create_node(
                "section",
                title,
                current_chapter,
                current_chapter.level + 1,
                context.metadata(),
                counter,
            )
            current_chapter.children.append(current_section)
            current_subsection = None
            current_container = None
            context.inside_chapter = True
            continue

        if event == "subsection":
            if current_chapter is None:
                current_chapter = create_node("chapter", chapter_title, root, 1, context.metadata(), counter)
                root.children.append(current_chapter)
            if current_section is None:
                current_section = create_node("section", "Introduction", current_chapter, current_chapter.level + 1, context.metadata(), counter)
                current_chapter.children.append(current_section)
                context.update_for_event("section", "Introduction")
            flush_paragraph()
            active_block = None
            context.update_for_event("subsection", title)
            current_subsection = create_node(
                "subsection",
                title,
                current_section,
                current_section.level + 1,
                context.metadata(),
                counter,
            )
            current_section.children.append(current_subsection)
            current_container = None
            context.inside_chapter = True
            continue

        if event in {"figure", "source", "activity", "discussion", "glossary", "table", "box", "timeline"}:
            if not context.inside_chapter and not (title and title.lower().startswith("introduction")):
                continue
            if current_chapter is None:
                current_chapter = create_node("chapter", chapter_title, root, 1, context.metadata(), counter)
                root.children.append(current_chapter)
            flush_paragraph()
            block_type = classify_block_type(title or normalized)
            active_block = create_block(block_type, title or normalized, content="")
            continue

        if event == "block_marker":
            flush_paragraph()
            active_block = None
            continue

        if event == "paragraph":
            # Ignore TOC/front-matter that appears before the chapter content
            if not context.inside_chapter:
                low_norm = normalized.lower()
                if low_norm.startswith("introduction"):
                    # create an Introduction section to hold blocks
                    if current_chapter is None:
                        current_chapter = create_node("chapter", chapter_title, root, 1, context.metadata(), counter)
                        root.children.append(current_chapter)
                    current_section = create_node("section", "Introduction", current_chapter, current_chapter.level + 1, context.metadata(), counter)
                    current_chapter.children.append(current_section)
                    context.update_for_event("section", "Introduction")
                    current_container = None
                    context.inside_chapter = True
                    # if the line after 'Introduction' contains text, treat it as paragraph
                    stripped_intro = re.sub(r"^introduction\s*:?", "", low_norm).strip()
                    if stripped_intro:
                        pending_paragraph_lines.append(stripped_intro)
                else:
                    # skip until we reach Introduction or numbered heading
                    continue
            pending_paragraph_lines.append(line)
            continue

    flush_paragraph()
    return root


def serialize_tree(root: Node) -> dict:
    return root.to_dict()


def count_nodes(root: Node) -> int:
    return 1 + sum(count_nodes(child) for child in root.children)


def build_chapter_trees(chapter_markdowns: list, chapter_titles: Optional[list] = None) -> list:
    trees = []
    for index, chapter_markdown in enumerate(chapter_markdowns):
        title = chapter_titles[index] if chapter_titles and index < len(chapter_titles) else f"Chapter {index + 1}"
        root = build_semantic_tree(chapter_markdown, chapter_title=title)
        trees.append({
            "chapter_title": title,
            "node_count": count_nodes(root),
            "tree": serialize_tree(root),
        })
    return trees


def save_chapter_trees(output_path: str, chapter_markdowns: list, chapter_titles: Optional[list] = None):
    trees = build_chapter_trees(chapter_markdowns, chapter_titles)
    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump({"chapters": trees}, outfile, indent=2, ensure_ascii=False)
    return trees
