from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .chunking import Chunk, chunk_documents
from .documents import load_documents
from .embeddings import EmbeddingModel, cosine


@dataclass(frozen=True)
class IndexedChunk:
    chunk: Chunk
    embedding: list[float]


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


class VectorIndex:
    def __init__(self, items: list[IndexedChunk]) -> None:
        self.items = items

    @classmethod
    def build(
        cls,
        data_root: Path | str,
        embedding_model: EmbeddingModel,
        chunk_size: int = 900,
        overlap: int = 120,
    ) -> "VectorIndex":
        documents = load_documents(data_root)
        chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
        embeddings = embedding_model.embed([chunk.text for chunk in chunks])
        return cls([IndexedChunk(chunk=chunk, embedding=embedding) for chunk, embedding in zip(chunks, embeddings)])

    def search(self, query: str, embedding_model: EmbeddingModel, top_k: int = 5) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_embedding = embedding_model.embed([query])[0]
        scored = [
            SearchResult(chunk=item.chunk, score=cosine(query_embedding, item.embedding))
            for item in self.items
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def save(self, path: Path | str) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "1.0",
            "items": [
                {"chunk": asdict(item.chunk), "embedding": item.embedding}
                for item in self.items
            ],
        }
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "VectorIndex":
        index_path = Path(path)
        if not index_path.is_file():
            raise FileNotFoundError(f"index does not exist: {index_path}")
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "1.0":
            raise ValueError("unsupported index schema")
        items = [
            IndexedChunk(chunk=Chunk(**record["chunk"]), embedding=record["embedding"])
            for record in payload["items"]
        ]
        return cls(items)
