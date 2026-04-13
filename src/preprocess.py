"""
Assemble per-sample STAR count files into a samples x genes matrix,
normalize / select HVGs, merge with clinical metadata.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"


def load_manifest() -> list[dict]:
    with open(RAW_DIR / "manifest.json", encoding="utf-8") as f:
        return json.load(f)


def _read_star_counts_tsv(path: Path) -> pd.Series:
    """Parse GDC STAR–Counts TSV; return gene_id -> unstranded count."""
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
        # Fallback: typical column order gene_id, gene_name, gene_type, unstranded, ...
        counts = pd.to_numeric(df.iloc[:, 3], errors="coerce").fillna(0)

    series = pd.Series(counts.values, index=gid.values, name="counts")
    series = series.groupby(level=0).sum()
    return series


def _find_count_file(expr_root: Path, file_id: str) -> Path | None:
    """Locate the expression file for a GDC file UUID after tar extract."""
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
    for path in expr_root.rglob("*.tsv.gz"):
        if file_id in path.parts or file_id in path.name:
            return path
    for path in expr_root.rglob("*.tsv"):
        if path.suffix == ".tsv" and file_id in path.parts:
            return path
    return None


def build_expression_matrix(manifest: list[dict], expr_root: Path) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for hit in manifest:
        file_id = hit["file_id"]
        case = hit.get("cases") or []
        if not case:
            print(f"Warning: no case for file {file_id}")
            continue
        case_id = case[0].get("submitter_id")
        if not case_id:
            continue

        path = _find_count_file(expr_root, file_id)
        if path is None:
            print(f"Warning: missing expression file for {file_id}")
            continue

        try:
            counts = _read_star_counts_tsv(path)
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: failed reading {path}: {exc}")
            continue

        counts.name = case_id
        rows.append(counts)

    if not rows:
        raise RuntimeError("No expression columns assembled — check extract path.")

    matrix = pd.concat(rows, axis=1).T
    matrix.index.name = "case_id"
    print(f"Expression matrix: {matrix.shape[0]} samples x {matrix.shape[1]} genes")
    return matrix


def filter_and_normalize(matrix: pd.DataFrame, n_hvg: int = 2000) -> pd.DataFrame:
    min_samples = max(1, int(0.1 * matrix.shape[0]))
    expressed = (matrix > 0).sum(axis=0)
    matrix = matrix.loc[:, expressed >= min_samples]
    print(f"After low-expression filter: {matrix.shape}")

    matrix_log = np.log1p(matrix.astype(float))
    variances = matrix_log.var(axis=0)
    top_genes = variances.nlargest(min(n_hvg, matrix_log.shape[1])).index
    matrix_hvg = matrix_log[top_genes]
    print(f"After HVG selection ({len(top_genes)} genes): {matrix_hvg.shape}")
    return matrix_hvg


def load_and_process_clinical() -> pd.DataFrame:
    with open(RAW_DIR / "clinical.json", encoding="utf-8") as f:
        cases = json.load(f)

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

    df = pd.DataFrame(rows).set_index("case_id")
    return df


def main() -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    expr_root = RAW_DIR / "expression_files"

    print("Building expression matrix...")
    matrix = build_expression_matrix(manifest, expr_root)
    matrix.to_parquet(PROC_DIR / "expression_raw.parquet")

    print("Filtering and normalizing...")
    matrix_hvg = filter_and_normalize(matrix)
    matrix_hvg.to_parquet(PROC_DIR / "expression_hvg.parquet")

    print("Clinical table...")
    clinical = load_and_process_clinical()
    clinical.to_parquet(PROC_DIR / "clinical.parquet")

    combined = matrix_hvg.join(clinical, how="inner")
    combined.to_parquet(PROC_DIR / "combined.parquet")
    print(f"Combined saved: {combined.shape} -> {PROC_DIR / 'combined.parquet'}")


if __name__ == "__main__":
    main()
