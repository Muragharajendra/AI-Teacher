from __future__ import annotations

import json
from pathlib import Path


def serialize_chunks(chunks: list, output_path: str | Path | None = None) -> list[dict]:
    payload = [chunk.to_dict() if hasattr(chunk, "to_dict") else chunk for chunk in chunks]
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
    return payload
