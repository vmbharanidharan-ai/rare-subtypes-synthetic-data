from __future__ import annotations

from datetime import datetime
from pathlib import Path

from rare_synth.config import RareSynthConfig
from rare_synth.data.cbioportal import download_study_snapshot
from rare_synth.data.gdc import download_project_dataset
from rare_synth.paths import ProjectPaths
from rare_synth.pipeline.preprocess import run_preprocess
from rare_synth.pipeline.train import run_benchmark_models, run_train
from rare_synth.pipeline.validate import run_validate


def create_run_id(cohort_name: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    safe = cohort_name.lower().replace(" ", "-")
    return f"{safe}-{ts}"


def run_dirs(paths: ProjectPaths, run_id: str) -> tuple[Path, Path, Path]:
    base = paths.root / "results" / "runs" / run_id
    synthetic_dir = base / "synthetic"
    figures_dir = base / "figures"
    reports_dir = base / "reports"
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    return synthetic_dir, figures_dir, reports_dir


def run_download(paths: ProjectPaths, config: RareSynthConfig) -> None:
    if config.data_source == "gdc":
        if not config.project_id:
            raise ValueError("data.project_id is required for gdc source")
        download_project_dataset(
            raw_dir=paths.raw_dir,
            project_id=config.project_id,
            workflow_type=config.workflow_type,
        )
        return

    if config.data_source == "cbioportal":
        if not config.cbioportal_study_id:
            raise ValueError("data.cbioportal_study_id is required for cbioportal source")
        download_study_snapshot(paths.raw_dir, config.cbioportal_study_id)
        return

    raise ValueError(f"Unsupported data source: {config.data_source}")


def run_cbioportal_ingest(paths: ProjectPaths, config: RareSynthConfig) -> Path:
    if not config.cbioportal_study_id:
        raise ValueError("data.cbioportal_study_id is required for cbioportal ingestion")
    return download_study_snapshot(paths.raw_dir, config.cbioportal_study_id)


def run_preprocessing(paths: ProjectPaths, config: RareSynthConfig) -> Path:
    return run_preprocess(
        raw_dir=paths.raw_dir,
        processed_dir=paths.processed_dir,
        n_hvg=config.n_hvg,
    )


def run_training(paths: ProjectPaths, config: RareSynthConfig, run_id: str) -> tuple[Path, Path]:
    combined_path = paths.processed_dir / "combined.parquet"
    synthetic_dir, _, _ = run_dirs(paths, run_id)
    return run_train(
        combined_path=combined_path,
        synthetic_dir=synthetic_dir,
        epochs=config.train_epochs,
        model_name=config.train_model,
        synthetic_multiplier=config.synthetic_multiplier,
        synthetic_min_rows=config.synthetic_min_rows,
    )


def run_model_benchmark(paths: ProjectPaths, config: RareSynthConfig, run_id: str) -> Path:
    combined_path = paths.processed_dir / "combined.parquet"
    synthetic_dir, _, _ = run_dirs(paths, run_id)
    return run_benchmark_models(
        combined_path=combined_path,
        synthetic_dir=synthetic_dir,
        epochs=config.train_epochs,
        models=config.baseline_models,
        synthetic_multiplier=config.synthetic_multiplier,
        synthetic_min_rows=config.synthetic_min_rows,
    )


def run_validation(paths: ProjectPaths, config: RareSynthConfig, run_id: str) -> tuple[dict, Path]:
    combined_path = paths.processed_dir / "combined.parquet"
    synthetic_dir, figures_dir, reports_dir = run_dirs(paths, run_id)
    synthetic_path = synthetic_dir / f"synthetic_{config.train_model}_{config.train_epochs}ep.parquet"
    metrics = run_validate(
        combined_path=combined_path,
        synthetic_path=synthetic_path,
        figures_dir=figures_dir,
        reports_dir=reports_dir,
        sample_ks_genes=config.sample_ks_genes,
        tstr_max_genes=config.tstr_max_genes,
    )
    return metrics, reports_dir / "validation_metrics.json"


def run_all(paths: ProjectPaths, config: RareSynthConfig, run_id: str) -> tuple[dict, Path, Path]:
    run_download(paths, config)
    run_preprocessing(paths, config)
    model_path, synthetic_path = run_training(paths, config, run_id=run_id)
    metrics, metrics_path = run_validation(paths, config, run_id=run_id)
    return metrics, model_path, metrics_path
