from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rare_synth.config import load_config
from rare_synth.paths import build_paths
from rare_synth.pipeline.compare import compare_runs
from rare_synth.pipeline.orchestrator import (
    create_run_id,
    run_all,
    run_download,
    run_preprocessing,
    run_training,
    run_validation,
)
from rare_synth.pipeline.registry import append_registry_row, load_registry

app = FastAPI(title="RareSynth API", version="0.2.0")


class GenerateRequest(BaseModel):
    config: str = "configs/default_uvm.yaml"
    root: str = "."
    command: str = "all"
    run_id: Optional[str] = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/runs")
def list_runs(root: str = ".") -> dict:
    paths = build_paths(Path(root))
    df = load_registry(paths.root)
    return {"count": int(len(df)), "rows": df.tail(50).to_dict(orient="records")}


@app.get("/compare")
def compare(root: str = ".", run_a: Optional[str] = None, run_b: Optional[str] = None) -> dict:
    try:
        return compare_runs(Path(root), run_a=run_a, run_b=run_b)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/generate")
def generate(req: GenerateRequest) -> dict:
    paths = build_paths(Path(req.root))
    config = load_config(req.config)
    run_id = req.run_id or create_run_id(config.cohort_name)

    if req.command not in {"download", "preprocess", "train", "validate", "all"}:
        raise HTTPException(status_code=400, detail="command must be one of download/preprocess/train/validate/all")

    if req.command == "download":
        run_download(paths, config)
        append_registry_row(paths.root, run_id, config.cohort_name, req.config, "download", "success")
        return {"run_id": run_id, "status": "download_complete"}

    if req.command == "preprocess":
        combined_path = run_preprocessing(paths, config)
        append_registry_row(
            paths.root,
            run_id,
            config.cohort_name,
            req.config,
            "preprocess",
            "success",
            notes=f"combined={combined_path}",
        )
        return {"run_id": run_id, "status": "preprocess_complete", "combined_path": str(combined_path)}

    if req.command == "train":
        model_path, synthetic_path = run_training(paths, config, run_id=run_id)
        append_registry_row(
            paths.root,
            run_id,
            config.cohort_name,
            req.config,
            "train",
            "success",
            model_path=str(model_path),
            synthetic_path=str(synthetic_path),
        )
        return {
            "run_id": run_id,
            "status": "train_complete",
            "model_path": str(model_path),
            "synthetic_path": str(synthetic_path),
        }

    if req.command == "validate":
        metrics, metrics_path = run_validation(paths, config, run_id=run_id)
        append_registry_row(
            paths.root,
            run_id,
            config.cohort_name,
            req.config,
            "validate",
            "success",
            metrics_path=str(metrics_path),
        )
        return {"run_id": run_id, "status": "validate_complete", "metrics": metrics}

    metrics, model_path, metrics_path = run_all(paths, config, run_id=run_id)
    synthetic_path = paths.root / "results" / "runs" / run_id / "synthetic" / f"synthetic_{config.train_epochs}ep.parquet"
    append_registry_row(
        paths.root,
        run_id,
        config.cohort_name,
        req.config,
        "all",
        "success",
        model_path=str(model_path),
        synthetic_path=str(synthetic_path),
        metrics_path=str(metrics_path),
    )
    return {
        "run_id": run_id,
        "status": "all_complete",
        "model_path": str(model_path),
        "synthetic_path": str(synthetic_path),
        "metrics": metrics,
    }
