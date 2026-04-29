from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RareSynthConfig:
    cohort_name: str
    data_source: str
    project_id: str
    workflow_type: str
    cbioportal_study_id: str | None
    n_hvg: int
    random_seed: int
    train_epochs: int
    train_model: str
    baseline_models: list[str]
    synthetic_multiplier: int
    synthetic_min_rows: int
    sample_ks_genes: int
    tstr_max_genes: int


def _require(d: dict[str, Any], key: str) -> Any:
    if key not in d:
        raise KeyError(f"Missing required config key: {key}")
    return d[key]


def load_config(path: str | Path) -> RareSynthConfig:
    cfg_path = Path(path)
    with open(cfg_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cohort = _require(raw, "cohort")
    data = _require(raw, "data")
    preprocessing = _require(raw, "preprocessing")
    train = _require(raw, "train")
    validate = _require(raw, "validate")

    source = str(_require(data, "source")).lower()
    if source not in {"gdc", "cbioportal"}:
        raise ValueError("data.source must be one of: gdc, cbioportal")

    return RareSynthConfig(
        cohort_name=str(_require(cohort, "name")),
        data_source=source,
        project_id=str(data.get("project_id", "")),
        workflow_type=str(data.get("workflow_type", "STAR - Counts")),
        cbioportal_study_id=data.get("cbioportal_study_id"),
        n_hvg=int(_require(preprocessing, "n_hvg")),
        random_seed=int(_require(preprocessing, "random_seed")),
        train_epochs=int(_require(train, "epochs")),
        train_model=str(train.get("model", "ctgan")).lower(),
        baseline_models=[str(x).lower() for x in train.get("baseline_models", ["ctgan", "tvae"])],
        synthetic_multiplier=int(_require(train, "synthetic_multiplier")),
        synthetic_min_rows=int(_require(train, "synthetic_min_rows")),
        sample_ks_genes=int(_require(validate, "sample_ks_genes")),
        tstr_max_genes=int(_require(validate, "tstr_max_genes")),
    )
