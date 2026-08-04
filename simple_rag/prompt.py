from __future__ import annotations

from dataclasses import dataclass

from .roles import RoleProfile
from .store import SearchResult


@dataclass(frozen=True)
class CompiledPrompt:
    messages: list[dict[str, str]]
    context_text: str
    sources: list[dict[str, str | float]]


class PromptBuilder:
    def __init__(self, max_context_chars: int = 5000) -> None:
        if max_context_chars < 500:
            raise ValueError("max_context_chars must be at least 500")
        self.max_context_chars = max_context_chars

    def build(self, query: str, role: RoleProfile, results: list[SearchResult]) -> CompiledPrompt:
        context_parts: list[str] = []
        sources: list[dict[str, str | float]] = []
        used_chars = 0
        for index, result in enumerate(results, start=1):
            source = result.chunk.source_path
            block = (
                f"[资料 {index}] source={source} span={result.chunk.start}:{result.chunk.end} "
                f"score={result.score:.4f}\n{result.chunk.text}"
            )
            if used_chars + len(block) > self.max_context_chars:
                continue
            context_parts.append(block)
            used_chars += len(block)
            sources.append(
                {
                    "source": source,
                    "start": result.chunk.start,
                    "end": result.chunk.end,
                    "score": round(result.score, 6),
                }
            )
        context_text = "\n\n".join(context_parts) or "没有检索到可用资料。"
        system = "\n\n".join(
            [
                role.render_system_rules(),
                "你必须优先依据<检索资料>回答。不要把检索资料之外的信息说成已证实事实。",
                "回答中需要保留角色口吻，但不能牺牲事实准确性。",
            ]
        )
        user = "\n".join(
            [
                "<检索资料>",
                context_text,
                "</检索资料>",
                "",
                "<用户问题>",
                query,
                "</用户问题>",
            ]
        )
        return CompiledPrompt(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            context_text=context_text,
            sources=sources,
        )
