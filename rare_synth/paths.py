from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    raw_dir: Path
    processed_dir: Path
    synthetic_dir: Path
    figures_dir: Path
    reports_dir: Path



def build_paths(root: str | Path | None = None) -> ProjectPaths:
    if root is None:
        root = Path(__file__).resolve().parent.parent
    root = Path(root).resolve()

    return ProjectPaths(
        root=root,
        raw_dir=root / "data" / "raw",
        processed_dir=root / "data" / "processed",
        synthetic_dir=root / "results" / "synthetic",
        figures_dir=root / "results" / "figures",
        reports_dir=root / "results" / "reports",
    )
