from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

CBIO_BASE = "https://www.cbioportal.org/api"


def _get(path: str, params: dict | None = None) -> requests.Response:
    headers = {"Accept": "application/json"}
    resp = requests.get(f"{CBIO_BASE}{path}", params=params, headers=headers, timeout=120)
    resp.raise_for_status()
    return resp


def fetch_samples(study_id: str) -> pd.DataFrame:
    resp = _get(f"/studies/{study_id}/samples")
    data = resp.json()
    return pd.DataFrame(data)


def fetch_patients(study_id: str) -> pd.DataFrame:
    resp = _get(f"/studies/{study_id}/patients")
    data = resp.json()
    return pd.DataFrame(data)


def fetch_molecular_profiles(study_id: str) -> pd.DataFrame:
    resp = _get(f"/studies/{study_id}/molecular-profiles")
    return pd.DataFrame(resp.json())


def fetch_clinical_data(study_id: str) -> pd.DataFrame:
    resp = _get(f"/studies/{study_id}/clinical-data")
    return pd.DataFrame(resp.json())


def download_study_snapshot(raw_dir: Path, study_id: str) -> Path:
    out_dir = raw_dir / "cbioportal" / study_id
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = fetch_samples(study_id)
    patients = fetch_patients(study_id)
    clinical = fetch_clinical_data(study_id)
    molecular_profiles = fetch_molecular_profiles(study_id)

    samples.to_csv(out_dir / "samples.csv", index=False)
    patients.to_csv(out_dir / "patients.csv", index=False)
    clinical.to_csv(out_dir / "clinical_data.csv", index=False)
    molecular_profiles.to_csv(out_dir / "molecular_profiles.csv", index=False)

    manifest = pd.DataFrame(
        {
            "study_id": [study_id],
            "n_samples": [len(samples)],
            "n_patients": [len(patients)],
            "n_clinical_rows": [len(clinical)],
            "n_molecular_profiles": [len(molecular_profiles)],
        }
    )
    manifest_path = out_dir / "snapshot_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"cBioPortal snapshot saved: {manifest_path}")
    return out_dir
