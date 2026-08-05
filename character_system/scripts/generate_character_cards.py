from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.validation import (  # noqa: E402
    load_json,
    load_jsonl,
    validate_cross_references,
    validate_events,
    validate_npc,
)
from runtime.paths import default_evidence_root  # noqa: E402


NOVEL_FILES = [f"{number}.txt" for number in range(1453, 1459)]


def build_extraction_request(npc_names: list[str], evidence_root: Path) -> str:
    prompt = (ROOT / "prompts" / "character_extraction.prompt").read_text(encoding="utf-8")
    parts = [
        prompt,
        "",
        "指定角色：" + "、".join(npc_names),
        "",
        "【带稳定行号的小说原文】",
    ]
    for filename in NOVEL_FILES:
        path = evidence_root / filename
        lines = path.read_text(encoding="utf-8").splitlines()
        parts.append(f"\n### {filename}")
        parts.extend(f"{filename}:L{index} {line}" for index, line in enumerate(lines, 1))
    return "\n".join(parts)


def build_report(evidence_root: Path) -> str:
    events = load_jsonl(ROOT / "story" / "story_events.jsonl")
    npcs = [load_json(path) for path in sorted((ROOT / "characters").glob("*.json"))]
    for npc in npcs:
        validate_npc(npc, evidence_root)
    validate_events(events, evidence_root)
    validate_cross_references(npcs, events)

    lines = [
        "# 角色卡人工检查报告",
        "",
        f"- 已校验角色：{len(npcs)}",
        f"- 公共事件：{len(events)}",
        "- 证据范围：1453.txt–1458.txt（稳定行号）",
        "- 推断处理：事件 source_type/confidence 与角色获知方式分别记录",
        "",
    ]
    for npc in npcs:
        identity = npc["identity_core"]
        lines.extend(
            [
                f"## {identity['name']}（{npc['npc_id']}）",
                "",
                f"- 世界位置：{identity['worldview_position']}",
                f"- 默认截止点：{npc['knowledge_scope']['story_cutoff']}",
                f"- 心理特征数：{len(identity['psychological_traits'])}",
                f"- 关系数：{len(npc['relationships'])}",
                f"- 情节记忆数：{len(npc['episodic_memory'])}",
                "",
                "### 核心心理特征与证据",
                "",
            ]
        )
        for index, trait in enumerate(identity["psychological_traits"]):
            refs = "、".join(
                npc["evidence_refs"][f"identity_core.psychological_traits[{index}]"]
            )
            lines.append(f"- {trait}（{refs}）")
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare and verify the card-generation workflow.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="Build a line-numbered extraction request")
    prepare.add_argument("--npc", nargs="+", default=["陆江仙", "玄谙"])
    prepare.add_argument("--output", type=Path)
    prepare.add_argument(
        "--evidence-root",
        type=Path,
        default=default_evidence_root(ROOT),
        help="Directory containing source evidence files such as 1453.txt",
    )
    report = subparsers.add_parser("report", help="Validate data and build the review report")
    report.add_argument("--output", type=Path)
    report.add_argument(
        "--evidence-root",
        type=Path,
        default=default_evidence_root(ROOT),
        help="Directory containing source evidence files such as 1453.txt",
    )
    return parser.parse_args()


def emit(text: str, output: Path | None) -> None:
    if output is None:
        print(text)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {output.resolve()}")


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        emit(build_extraction_request(args.npc, args.evidence_root), args.output)
    else:
        emit(build_report(args.evidence_root), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
