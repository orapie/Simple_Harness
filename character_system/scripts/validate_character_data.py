from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate character-system data.")
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=default_evidence_root(ROOT),
        help="Directory containing source evidence files such as 1453.txt",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evidence_root = args.evidence_root
    npc_paths = sorted((ROOT / "characters").glob("*.json"))
    npcs = [load_json(path) for path in npc_paths]
    events = load_jsonl(ROOT / "story" / "story_events.jsonl")
    for npc in npcs:
        validate_npc(npc, evidence_root)
    validate_events(events, evidence_root)
    validate_cross_references(npcs, events)
    print(f"OK: {len(npcs)} NPC cards, {len(events)} shared events, all references valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
