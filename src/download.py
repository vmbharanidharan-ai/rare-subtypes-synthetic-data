"""
Download open-access TCGA-UVM RNA-seq (STAR – Counts) + case metadata from GDC.
No API key required for open-access files.
"""

from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path

import requests

GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"
GDC_CASES_ENDPOINT = "https://api.gdc.cancer.gov/cases"

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def get_uvm_rnaseq_file_ids() -> list[dict]:
    """Query GDC for TCGA-UVM RNA-seq gene expression (STAR counts), open access."""
    filters = {
        "op": "and",
        "content": [
            {
                "op": "=",
                "content": {"field": "cases.project.project_id", "value": ["TCGA-UVM"]},
            },
            {
                "op": "=",
                "content": {"field": "files.experimental_strategy", "value": ["RNA-Seq"]},
            },
            {
                "op": "=",
                "content": {
                    "field": "files.data_type",
                    "value": ["Gene Expression Quantification"],
                },
            },
            {
                "op": "=",
                "content": {
                    "field": "files.analysis.workflow_type",
                    "value": ["STAR - Counts"],
                },
            },
            {"op": "=", "content": {"field": "files.access", "value": ["open"]}},
        ],
    }

    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,cases.submitter_id,cases.samples.sample_type",
        "format": "JSON",
        "size": "500",
    }

    response = requests.get(GDC_FILES_ENDPOINT, params=params, timeout=120)
    response.raise_for_status()
    payload = response.json()
    hits = payload.get("data", {}).get("hits", [])
    print(f"Found {len(hits)} TCGA-UVM STAR-count RNA-seq files (open)")
    return hits


def download_data_tar(file_ids: list[str], output_path: Path) -> None:
    """POST /data to retrieve a tar archive of the given file UUIDs."""
    payload = {"ids": file_ids}
    response = requests.post(
        GDC_DATA_ENDPOINT,
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"},
        stream=True,
        timeout=600,
    )
    response.raise_for_status()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    print(f"Saved archive to {output_path}")


def extract_gdc_tar(archive_path: Path, dest_dir: Path) -> None:
    """GDC returns a tar archive; extract into dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r") as tar:
        tar.extractall(path=dest_dir)
    print(f"Extracted to {dest_dir}")


def download_clinical_data() -> list[dict]:
    """Pull case-level fields for TCGA-UVM (open metadata)."""
    filters = {
        "op": "=",
        "content": {"field": "project.project_id", "value": ["TCGA-UVM"]},
    }
    fields = [
        "submitter_id",
        "diagnoses.age_at_diagnosis",
        "diagnoses.vital_status",
        "diagnoses.days_to_death",
        "diagnoses.days_to_last_follow_up",
        "diagnoses.tumor_stage",
        "demographic.gender",
        "demographic.race",
    ]
    params = {
        "filters": json.dumps(filters),
        "fields": ",".join(fields),
        "format": "JSON",
        "size": "500",
    }
    response = requests.get(GDC_CASES_ENDPOINT, params=params, timeout=120)
    response.raise_for_status()
    cases = response.json().get("data", {}).get("hits", [])
    print(f"Downloaded metadata for {len(cases)} cases")
    return cases


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    hits = get_uvm_rnaseq_file_ids()
    if not hits:
        raise SystemExit("No files returned from GDC — check filters or network.")

    manifest_path = RAW_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(hits, f, indent=2)

    file_ids = [h["file_id"] for h in hits]
    archive_path = RAW_DIR / "uvm_rnaseq.tar"
    download_data_tar(file_ids, archive_path)

    expr_root = RAW_DIR / "expression_files"
    if expr_root.exists():
        shutil.rmtree(expr_root)
    extract_gdc_tar(archive_path, expr_root)

    cases = download_clinical_data()
    with open(RAW_DIR / "clinical.json", "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2)

    print("Done. See data/raw/manifest.json, expression_files/, clinical.json")


if __name__ == "__main__":
    main()
