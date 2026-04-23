from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_manifest(raw_dir: Path) -> list[dict]:
    with open(raw_dir / "manifest.json", encoding="utf-8") as f:
        return json.load(f)


def _read_star_counts_tsv(path: Path) -> pd.Series:
    opener = gzip.open if str(path).endswith(".gz") else open
    mode = "rt" if str(path).endswith(".gz") else "r"
    with opener(path, mode, encoding="utf-8", errors="replace") as handle:
        df = pd.read_csv(handle, sep="\t", comment="#", low_memory=False)

    if "gene_id" in df.columns:
        gid = df["gene_id"].astype(str)
    else:
        gid = df.iloc[:, 0].astype(str)

    if "unstranded" in df.columns:
        counts = pd.to_numeric(df["unstranded"], errors="coerce").fillna(0)
    else:
        counts = pd.to_numeric(df.iloc[:, 3], errors="coerce").fillna(0)

    return pd.Series(counts.values, index=gid.values, name="counts").groupby(level=0).sum()


def _find_count_file(expr_root: Path, file_id: str) -> Path | None:
    direct = expr_root / file_id
    if direct.is_dir():
        for pattern in ("*.tsv.gz", "*.tsv"):
            found = list(direct.glob(pattern))
            if found:
                return found[0]

    for pattern in (f"{file_id}*.tsv.gz", f"{file_id}*.tsv"):
        found = list(expr_root.rglob(pattern))
        if found:
            return found[0]

    return None


def build_expression_matrix(manifest: list[dict], expr_root: Path) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for hit in manifest:
        file_id = hit["file_id"]
        case = hit.get("cases") or []
        if not case:
            continue

        case_id = case[0].get("submitter_id")
        if not case_id:
            continue

        path = _find_count_file(expr_root, file_id)
        if path is None:
            continue

        counts = _read_star_counts_tsv(path)
        counts.name = case_id
        rows.append(counts)

    if not rows:
        raise RuntimeError("No expression files assembled.")

    matrix = pd.concat(rows, axis=1).T
    matrix.index.name = "case_id"
    return matrix


def filter_and_normalize(matrix: pd.DataFrame, n_hvg: int) -> pd.DataFrame:
    min_samples = max(1, int(0.1 * matrix.shape[0]))
    matrix = matrix.loc[:, (matrix > 0).sum(axis=0) >= min_samples]

    matrix_log = np.log1p(matrix.astype(float))
    top_genes = matrix_log.var(axis=0).nlargest(min(n_hvg, matrix_log.shape[1])).index
    return matrix_log[top_genes]


def load_clinical_table(raw_dir: Path) -> pd.DataFrame:
    with open(raw_dir / "clinical.json", encoding="utf-8") as handle:
        cases = json.load(handle)

    rows = []
    for case in cases:
        case_id = case.get("submitter_id")
        if not case_id:
            continue
        diag = (case.get("diagnoses") or [{}])[0]
        dem = case.get("demographic") or {}
        rows.append(
            {
                "case_id": case_id,
                "age_at_diagnosis": diag.get("age_at_diagnosis", np.nan),
                "vital_status": diag.get("vital_status") or "unknown",
                "days_to_death": diag.get("days_to_death", np.nan),
                "days_to_last_follow_up": diag.get("days_to_last_follow_up", np.nan),
                "tumor_stage": diag.get("tumor_stage") or "unknown",
                "gender": dem.get("gender") or "unknown",
                "race": dem.get("race") or "unknown",
            }
        )

    return pd.DataFrame(rows).set_index("case_id")


def run_preprocess(raw_dir: Path, processed_dir: Path, n_hvg: int) -> Path:
    processed_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = raw_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} does not exist. Run download first.")

    manifest = load_manifest(raw_dir)
    expression = build_expression_matrix(manifest, raw_dir / "expression_files")
    expression.to_parquet(processed_dir / "expression_raw.parquet")

    expression_hvg = filter_and_normalize(expression, n_hvg=n_hvg)
    expression_hvg.to_parquet(processed_dir / "expression_hvg.parquet")

    clinical = load_clinical_table(raw_dir)
    clinical.to_parquet(processed_dir / "clinical.parquet")

    combined = expression_hvg.join(clinical, how="inner")
    combined_path = processed_dir / "combined.parquet"
    combined.to_parquet(combined_path)
    return combined_path
