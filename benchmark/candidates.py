"""Candidate discovery helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Set


def discover_run_logdirs(candidate_dir: Path) -> List[Path]:
    """Find run-like logdirs under a candidate root, following directory symlinks."""
    if not candidate_dir.exists():
        raise ValueError(f"{candidate_dir}: candidate directory does not exist")
    if not candidate_dir.is_dir():
        raise ValueError(f"{candidate_dir}: candidate path is not a directory")

    logdirs = []
    seen_dirs: Set[Path] = set()
    seen_logdirs: Set[Path] = set()
    stack = [candidate_dir]
    while stack:
        current = stack.pop()
        try:
            resolved = current.resolve(strict=True)
        except OSError:
            continue
        if resolved in seen_dirs:
            continue
        seen_dirs.add(resolved)

        if (current / "parameters.pkl").is_file():
            if resolved not in seen_logdirs:
                logdirs.append(current)
                seen_logdirs.add(resolved)

        try:
            children: Iterable[Path] = sorted(current.iterdir())
        except OSError:
            continue
        for child in reversed(list(children)):
            if child.is_dir():
                stack.append(child)

    return sorted(logdirs, key=lambda path: str(path))
