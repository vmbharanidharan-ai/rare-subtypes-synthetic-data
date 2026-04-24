from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
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


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(root: str = ".") -> str:
    paths = build_paths(Path(root))
    df = load_registry(paths.root)
    rows = df.tail(50).to_dict(orient="records")
    table_rows = "".join(
        [
            "<tr>"
            f"<td>{r.get('timestamp_utc','')}</td>"
            f"<td>{r.get('run_id','')}</td>"
            f"<td>{r.get('cohort_name','')}</td>"
            f"<td>{r.get('stage','')}</td>"
            f"<td>{r.get('status','')}</td>"
            f"<td>{r.get('notes','')}</td>"
            "</tr>"
            for r in rows
        ]
    )
    return f"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8"/>
    <title>RareSynth Dashboard</title>
    <style>
      body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 24px; }}
      h1 {{ margin-bottom: 4px; }}
      .sub {{ color: #666; margin-bottom: 16px; }}
      table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
      th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }}
      th {{ background: #f7f7f7; }}
      code {{ background: #f2f2f2; padding: 1px 4px; border-radius: 3px; }}
    </style>
  </head>
  <body>
    <h1>RareSynth Dashboard</h1>
    <div class="sub">Recent run registry entries ({len(rows)} shown)</div>
    <p>
      Endpoints:
      <a href="/health">/health</a> |
      <a href="/runs?root=.">/runs</a> |
      <a href="/compare?root=.">/compare</a>
    </p>
    <table>
      <thead>
        <tr>
          <th>Timestamp (UTC)</th>
          <th>Run ID</th>
          <th>Cohort</th>
          <th>Stage</th>
          <th>Status</th>
          <th>Notes</th>
        </tr>
      </thead>
      <tbody>{table_rows}</tbody>
    </table>
  </body>
</html>
"""


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
