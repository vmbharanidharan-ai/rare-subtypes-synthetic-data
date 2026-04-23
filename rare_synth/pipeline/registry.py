from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


REGISTRY_COLUMNS = [
    "timestamp_utc",
    "run_id",
    "cohort_name",
    "config_path",
    "stage",
    "status",
    "model_path",
    "synthetic_path",
    "metrics_path",
    "notes",
]


def registry_path(root: Path) -> Path:
    return root / "results" / "runs" / "index.csv"


def append_registry_row(
    root: Path,
    run_id: str,
    cohort_name: str,
    config_path: str,
    stage: str,
    status: str,
    model_path: str | None = None,
    synthetic_path: str | None = None,
    metrics_path: str | None = None,
    notes: str | None = None,
) -> Path:
    path = registry_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "run_id": run_id,
        "cohort_name": cohort_name,
        "config_path": config_path,
        "stage": stage,
        "status": status,
        "model_path": model_path or "",
        "synthetic_path": synthetic_path or "",
        "metrics_path": metrics_path or "",
        "notes": notes or "",
    }

    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=REGISTRY_COLUMNS)

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)
    return path


def load_registry(root: Path) -> pd.DataFrame:
    path = registry_path(root)
    if not path.exists():
        return pd.DataFrame(columns=REGISTRY_COLUMNS)
    return pd.read_csv(path)
