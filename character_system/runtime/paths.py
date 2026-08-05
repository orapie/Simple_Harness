from __future__ import annotations

from pathlib import Path


def default_evidence_root(character_root: Path | str) -> Path:
    root = Path(character_root).resolve()
    bundled_data = root.parent / "data" / "novel_test"
    if bundled_data.is_dir():
        return bundled_data
    return root.parent
