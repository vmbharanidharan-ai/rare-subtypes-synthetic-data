from __future__ import annotations

import json
import shutil
import tarfile
import time
from pathlib import Path

import requests

GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"
GDC_CASES_ENDPOINT = "https://api.gdc.cancer.gov/cases"


def get_rnaseq_file_hits(project_id: str, workflow_type: str) -> list[dict]:
    filters = {
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "cases.project.project_id", "value": [project_id]}},
            {"op": "=", "content": {"field": "files.experimental_strategy", "value": ["RNA-Seq"]}},
            {"op": "=", "content": {"field": "files.data_type", "value": ["Gene Expression Quantification"]}},
            {"op": "=", "content": {"field": "files.analysis.workflow_type", "value": [workflow_type]}},
            {"op": "=", "content": {"field": "files.access", "value": ["open"]}},
        ],
    }

    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,cases.submitter_id,cases.samples.sample_type",
        "format": "JSON",
        "size": "1000",
    }

    response = requests.get(GDC_FILES_ENDPOINT, params=params, timeout=120)
    response.raise_for_status()
    hits = response.json().get("data", {}).get("hits", [])
    print(f"Found {len(hits)} open RNA-seq files for {project_id}")
    return hits


def download_data_tar(file_ids: list[str], output_path: Path) -> None:
    payload = {"ids": file_ids}
    response = requests.post(
        GDC_DATA_ENDPOINT,
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"},
        stream=True,
        timeout=1200,
    )
    response.raise_for_status()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)


def validate_tar(archive_path: Path) -> None:
    """Fail fast when archive is truncated/corrupt."""
    if not archive_path.exists() or archive_path.stat().st_size == 0:
        raise RuntimeError(f"Archive missing/empty: {archive_path}")
    with tarfile.open(archive_path, "r") as tar:
        # Iterate all headers to trigger CRC/EOF issues before extraction.
        for _ in tar:
            pass


def extract_tar(archive_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r") as tar:
        tar.extractall(path=dest_dir)


def download_clinical_data(project_id: str) -> list[dict]:
    filters = {"op": "=", "content": {"field": "project.project_id", "value": [project_id]}}
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
        "size": "1000",
    }

    response = requests.get(GDC_CASES_ENDPOINT, params=params, timeout=120)
    response.raise_for_status()
    return response.json().get("data", {}).get("hits", [])


def download_project_dataset(raw_dir: Path, project_id: str, workflow_type: str) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)

    hits = get_rnaseq_file_hits(project_id=project_id, workflow_type=workflow_type)
    if not hits:
        raise SystemExit(f"No files returned for {project_id}. Check configuration/network.")

    with open(raw_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(hits, handle, indent=2)

    archive_path = raw_dir / f"{project_id.lower()}_rnaseq.tar"
    file_ids = [item["file_id"] for item in hits]

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            if archive_path.exists():
                archive_path.unlink()
            download_data_tar(file_ids, archive_path)
            validate_tar(archive_path)
            break
        except (requests.RequestException, tarfile.TarError, EOFError, RuntimeError) as exc:
            if attempt == max_attempts:
                raise RuntimeError(
                    f"Failed to download/validate {project_id} archive after {max_attempts} attempts"
                ) from exc
            backoff_seconds = 3 * attempt
            print(
                f"Download attempt {attempt}/{max_attempts} failed ({exc}). "
                f"Retrying in {backoff_seconds}s..."
            )
            time.sleep(backoff_seconds)

    expr_root = raw_dir / "expression_files"
    if expr_root.exists():
        shutil.rmtree(expr_root)
    extract_tar(archive_path, expr_root)

    clinical = download_clinical_data(project_id=project_id)
    with open(raw_dir / "clinical.json", "w", encoding="utf-8") as handle:
        json.dump(clinical, handle, indent=2)

    print(f"Download complete for {project_id}: {len(hits)} expression files")
