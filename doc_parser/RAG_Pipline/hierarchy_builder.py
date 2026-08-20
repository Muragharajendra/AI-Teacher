from .parser import build_chapter_trees, build_semantic_tree, save_chapter_trees
from .semantic_chunker import HierarchyChunker, serialize_chunks


def build_semantic_chunks(root):
    return HierarchyChunker().build(root)


def export_semantic_chunks(chunks, output_path=None):
    return serialize_chunks(chunks, output_path)


__all__ = [
    "build_semantic_tree",
    "build_chapter_trees",
    "save_chapter_trees",
    "build_semantic_chunks",
    "export_semantic_chunks",
    "HierarchyChunker",
]
