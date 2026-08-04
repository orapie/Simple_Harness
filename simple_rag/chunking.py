from __future__ import annotations

from dataclasses import dataclass

from .documents import Document


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source_path: str
    start: int
    end: int
    metadata: dict[str, str]


def chunk_documents(
    documents: list[Document], chunk_size: int = 900, overlap: int = 120
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    chunks: list[Chunk] = []
    for document in documents:
        text = _normalize_text(document.text)
        if not text:
            continue
        start = 0
        doc_index = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                boundary = max(text.rfind("\n", start, end), text.rfind("。", start, end))
                if boundary > start + chunk_size // 2:
                    end = boundary + 1
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    Chunk(
                        id=f"{document.path}#{doc_index}",
                        text=chunk_text,
                        source_path=document.path,
                        start=start,
                        end=end,
                        metadata=document.metadata,
                    )
                )
                doc_index += 1
            if end >= len(text):
                break
            start = max(0, end - overlap)
    return chunks


def _normalize_text(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()
