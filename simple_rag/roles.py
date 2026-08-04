from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RoleProfile:
    id: str
    name: str
    persona: str
    style: list[str] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)
    refusal: str = "如果检索资料不足，明确说明资料不足，不要编造。"

    @classmethod
    def load(cls, path: Path | str) -> "RoleProfile":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            id=payload["id"],
            name=payload["name"],
            persona=payload["persona"],
            style=list(payload.get("style", [])),
            rules=list(payload.get("rules", [])),
            refusal=payload.get("refusal", "如果检索资料不足，明确说明资料不足，不要编造。"),
        )

    def render_system_rules(self) -> str:
        lines = [
            f"你正在扮演：{self.name}",
            f"角色设定：{self.persona}",
        ]
        if self.style:
            lines.append("表达风格：")
            lines.extend(f"- {item}" for item in self.style)
        if self.rules:
            lines.append("行为规则：")
            lines.extend(f"- {item}" for item in self.rules)
        lines.append(f"资料不足时：{self.refusal}")
        return "\n".join(lines)
