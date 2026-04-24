from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from rare_synth.pipeline.registry import load_registry


KEY_METRICS = ["tstr_auc", "trtr_auc", "privacy_proxy_auc", "nn_leakage_ratio", "ks_pct_below_0_2"]


def _latest_unique_run_ids(df: pd.DataFrame, n: int = 2) -> list[str]:
    if df.empty:
        return []
    ordered = []
    seen = set()
    for run_id in reversed(df["run_id"].tolist()):
        if run_id in seen:
            continue
        seen.add(run_id)
        ordered.append(run_id)
        if len(ordered) >= n:
            break
    return list(reversed(ordered))


def _metrics_for_run(root: Path, run_id: str) -> dict:
    metrics_path = root / "results" / "runs" / run_id / "reports" / "validation_metrics.json"
    if not metrics_path.exists():
        return {"run_id": run_id, "missing_metrics": True}

    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)

    out = {"run_id": run_id, "missing_metrics": False}
    for key in KEY_METRICS:
        out[key] = metrics.get(key)
    return out


def compare_runs(root: Path, run_a: str | None = None, run_b: str | None = None) -> dict:
    registry = load_registry(root)

    if run_a is None or run_b is None:
        latest = _latest_unique_run_ids(registry, n=2)
        if len(latest) < 2:
            raise ValueError("Need at least two runs in registry to compare.")
        run_a, run_b = latest[0], latest[1]

    a = _metrics_for_run(root, run_a)
    b = _metrics_for_run(root, run_b)

    deltas = {}
    for key in KEY_METRICS:
        av = a.get(key)
        bv = b.get(key)
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            deltas[key] = bv - av
        else:
            deltas[key] = None

    return {"run_a": a, "run_b": b, "delta_run_b_minus_run_a": deltas}
