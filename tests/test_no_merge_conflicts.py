"""Guardrail test: fail fast if merge-conflict markers are present."""

from __future__ import annotations

from pathlib import Path


CONFLICT_MARKERS = ("<<<<<<< ", "=======", ">>>>>>> ")
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache"}


def test_no_unresolved_merge_conflict_markers():
    root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue  # Binary/non-UTF8 file, skip.

        lines = text.splitlines()
        for idx, line in enumerate(lines, start=1):
            if line.startswith(CONFLICT_MARKERS):
                rel = path.relative_to(root)
                offenders.append(f"{rel}:{idx}: {line[:12]}")

    assert not offenders, "Unresolved merge conflict markers found:\n" + "\n".join(offenders)

