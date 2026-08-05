from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from character_system.runtime import PromptCompiler, RuntimeContext
from character_system.runtime.paths import default_evidence_root

from .embeddings import EmbeddingModel
from .llm import GenerationOptions, LocalLLM
from .store import SearchResult, VectorIndex


@dataclass(frozen=True)
class CharacterRagPrompt:
    messages: list[dict[str, str]]
    sources: list[dict[str, str | float]]
    character_debug: dict[str, Any]
    context_text: str


@dataclass(frozen=True)
class CharacterRagResponse:
    answer: str
    prompt: CharacterRagPrompt
    results: list[SearchResult]


class CharacterRagPromptBuilder:
    def __init__(self, max_context_chars: int = 5000) -> None:
        if max_context_chars < 500:
            raise ValueError("max_context_chars must be at least 500")
        self.max_context_chars = max_context_chars

    def build(
        self,
        character_messages: list[dict[str, str]],
        user_input: str,
        results: list[SearchResult],
        character_debug: dict[str, Any] | None = None,
    ) -> CharacterRagPrompt:
        if len(character_messages) != 2:
            raise ValueError("character compiler must return exactly system and user messages")
        if character_messages[0]["role"] != "system" or character_messages[1]["role"] != "user":
            raise ValueError("character messages must be ordered as system, user")

        context_text, sources = self._compile_context(results)
        system = "\n\n".join(
            [
                character_messages[0]["content"],
                "【外部检索资料使用规则】\n"
                "外部检索资料是本轮运行时递交给角色参考的资料，不属于角色亲历剧情或长期记忆。"
                "可以依据这些资料回答现实、新闻、文档或背景问题，但必须区分“角色已知剧情”和“本轮外部资料”。"
                "外部资料不足时直接说明资料不足，不要编造来源。",
            ]
        )
        user = "\n".join(
            [
                "<外部检索资料>",
                context_text,
                "</外部检索资料>",
                "",
                "<玩家问题>",
                user_input,
                "</玩家问题>",
            ]
        )
        return CharacterRagPrompt(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            sources=sources,
            character_debug=character_debug or {},
            context_text=context_text,
        )

    def _compile_context(self, results: list[SearchResult]) -> tuple[str, list[dict[str, str | float]]]:
        parts: list[str] = []
        sources: list[dict[str, str | float]] = []
        used_chars = 0
        for index, result in enumerate(results, start=1):
            block = (
                f"[资料 {index}] source={result.chunk.source_path} "
                f"span={result.chunk.start}:{result.chunk.end} "
                f"score={result.score:.4f}\n{result.chunk.text}"
            )
            if used_chars + len(block) > self.max_context_chars:
                continue
            parts.append(block)
            used_chars += len(block)
            sources.append(
                {
                    "source": result.chunk.source_path,
                    "start": result.chunk.start,
                    "end": result.chunk.end,
                    "score": round(result.score, 6),
                }
            )
        return "\n\n".join(parts) or "没有检索到可用资料。", sources


class CharacterRagPipeline:
    def __init__(
        self,
        index: VectorIndex,
        embedding_model: EmbeddingModel,
        llm: LocalLLM,
        prompt_builder: CharacterRagPromptBuilder | None = None,
        character_root: Path | str = "character_system",
        evidence_root: Path | str | None = None,
    ) -> None:
        self.index = index
        self.embedding_model = embedding_model
        self.llm = llm
        self.prompt_builder = prompt_builder or CharacterRagPromptBuilder()
        root = Path(character_root)
        self.compiler = PromptCompiler(
            root,
            evidence_root=evidence_root or default_evidence_root(root),
        )

    @classmethod
    def from_index_file(
        cls,
        index_path: Path | str,
        embedding_model: EmbeddingModel,
        llm: LocalLLM,
        prompt_builder: CharacterRagPromptBuilder | None = None,
        character_root: Path | str = "character_system",
        evidence_root: Path | str | None = None,
    ) -> "CharacterRagPipeline":
        return cls(
            VectorIndex.load(index_path),
            embedding_model,
            llm,
            prompt_builder=prompt_builder,
            character_root=character_root,
            evidence_root=evidence_root,
        )

    def answer(
        self,
        query: str,
        npc_id: str,
        story_cutoff: str | None = None,
        top_k: int = 5,
        max_character_chars: int = 4500,
        options: GenerationOptions | None = None,
    ) -> CharacterRagResponse:
        results = self.index.search(query, self.embedding_model, top_k=top_k)
        compiled = self.compiler.build_npc_prompt(
            npc_id,
            query,
            RuntimeContext(story_cutoff=story_cutoff, max_chars=max_character_chars),
        )
        prompt = self.prompt_builder.build(
            compiled.messages,
            query,
            results,
            character_debug=compiled.debug,
        )
        answer = self.llm.generate(prompt.messages, options or GenerationOptions())
        return CharacterRagResponse(answer=answer, prompt=prompt, results=results)
