from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .embeddings import EmbeddingModel
from .llm import GenerationOptions, LocalLLM
from .prompt import CompiledPrompt, PromptBuilder
from .roles import RoleProfile
from .store import SearchResult, VectorIndex


@dataclass(frozen=True)
class RagResponse:
    answer: str
    prompt: CompiledPrompt
    results: list[SearchResult]


class RagPipeline:
    def __init__(
        self,
        index: VectorIndex,
        embedding_model: EmbeddingModel,
        llm: LocalLLM,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        self.index = index
        self.embedding_model = embedding_model
        self.llm = llm
        self.prompt_builder = prompt_builder or PromptBuilder()

    @classmethod
    def from_index_file(
        cls,
        index_path: Path | str,
        embedding_model: EmbeddingModel,
        llm: LocalLLM,
        prompt_builder: PromptBuilder | None = None,
    ) -> "RagPipeline":
        return cls(VectorIndex.load(index_path), embedding_model, llm, prompt_builder)

    def answer(
        self,
        query: str,
        role: RoleProfile,
        top_k: int = 5,
        options: GenerationOptions | None = None,
    ) -> RagResponse:
        results = self.index.search(query, self.embedding_model, top_k=top_k)
        prompt = self.prompt_builder.build(query, role, results)
        answer = self.llm.generate(prompt.messages, options or GenerationOptions())
        return RagResponse(answer=answer, prompt=prompt, results=results)
