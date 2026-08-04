from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass


class EmbeddingModel(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


@dataclass
class HashEmbeddingModel(EmbeddingModel):
    dimensions: int = 384
    ngram_min: int = 2
    ngram_max: int = 4

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        normalized = "".join(text.lower().split())
        if not normalized:
            return vector
        for token in self._tokens(normalized):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        return normalize(vector)

    def _tokens(self, text: str) -> list[str]:
        tokens: list[str] = []
        for size in range(self.ngram_min, self.ngram_max + 1):
            if len(text) < size:
                continue
            tokens.extend(text[index : index + size] for index in range(len(text) - size + 1))
        return tokens or [text]


class TransformersEmbeddingModel(EmbeddingModel):
    def __init__(
        self,
        model_name_or_path: str,
        device: str | None = None,
        batch_size: int = 8,
        max_length: int = 512,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers embeddings require: python -m pip install torch transformers"
            ) from exc

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.batch_size = batch_size
        self.max_length = max_length

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with self.torch.no_grad():
                output = self.model(**encoded)
            embeddings = self._mean_pool(output.last_hidden_state, encoded["attention_mask"])
            embeddings = self.torch.nn.functional.normalize(embeddings, p=2, dim=1)
            vectors.extend(embeddings.cpu().tolist())
        return vectors

    def _mean_pool(self, hidden_states, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        summed = (hidden_states * mask).sum(1)
        counts = mask.sum(1).clamp(min=1e-9)
        return summed / counts


def normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def make_embedding_model(model_name_or_path: str | None, device: str | None = None) -> EmbeddingModel:
    if model_name_or_path:
        return TransformersEmbeddingModel(model_name_or_path, device=device)
    return HashEmbeddingModel()
