from __future__ import annotations

from pathlib import Path
from typing import Optional
import json

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
    run_model_benchmark,
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


def _read_metrics_for_run(root: Path, run_id: str) -> dict:
    p = root / "results" / "runs" / run_id / "reports" / "validation_metrics.json"
    if not p.exists():
        return {"run_id": run_id, "has_metrics": False}
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    data["run_id"] = run_id
    data["has_metrics"] = True
    return data


def _quality_summary(metrics: dict) -> dict:
    if not metrics.get("has_metrics"):
        return {"grade": "N/A", "notes": ["No validation metrics found for this run."]}
    notes = []
    score = 0
    privacy_auc = metrics.get("privacy_proxy_auc")
    if isinstance(privacy_auc, (int, float)):
        if privacy_auc <= 0.7:
            score += 1
            notes.append("Privacy proxy looks acceptable.")
        elif privacy_auc <= 0.85:
            notes.append("Privacy separability is moderate; tune model.")
        else:
            notes.append("Privacy separability is high; likely synthetic-real mismatch.")
    ks = metrics.get("ks_pct_below_0_2")
    if isinstance(ks, (int, float)):
        if ks >= 40:
            score += 1
            notes.append("Distribution fidelity is strong.")
        elif ks >= 20:
            notes.append("Distribution fidelity is moderate.")
        else:
            notes.append("Distribution fidelity is weak.")
    nn = metrics.get("nn_leakage_ratio")
    if isinstance(nn, (int, float)):
        if nn <= 0.1:
            score += 1
            notes.append("Nearest-neighbor leakage risk appears low.")
        elif nn <= 0.25:
            notes.append("Nearest-neighbor leakage risk is moderate.")
        else:
            notes.append("Nearest-neighbor leakage risk is elevated.")

    grade = "Low"
    if score >= 3:
        grade = "High"
    elif score == 2:
        grade = "Medium"
    return {"grade": grade, "notes": notes}


@app.get("/compare")
def compare(root: str = ".", run_a: Optional[str] = None, run_b: Optional[str] = None) -> dict:
    try:
        return compare_runs(Path(root), run_a=run_a, run_b=run_b)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/run-summary")
def run_summary(root: str = ".", run_id: str = "") -> dict:
    if not run_id:
        raise HTTPException(status_code=400, detail="run_id is required")
    metrics = _read_metrics_for_run(Path(root), run_id)
    return {"metrics": metrics, "quality": _quality_summary(metrics)}


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


@app.get("/app", response_class=HTMLResponse)
def app_ui(root: str = ".") -> str:
    return f"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8"/>
    <title>RareSynth App</title>
    <style>
      body {{ font-family: -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; margin: 24px; color:#1a1a1a; }}
      h1 {{ margin: 0 0 4px 0; }}
      .sub {{ color:#666; margin-bottom:16px; }}
      .grid {{ display:grid; grid-template-columns: 1fr 1fr; gap:16px; }}
      .card {{ border:1px solid #ddd; border-radius:8px; padding:12px; }}
      label {{ display:block; font-size:12px; color:#444; margin-bottom:4px; }}
      input, select, button, textarea {{ width:100%; padding:8px; margin-bottom:8px; box-sizing:border-box; }}
      button {{ cursor:pointer; }}
      pre {{ background:#f7f7f7; padding:8px; overflow:auto; border-radius:6px; }}
      .mono {{ font-family: ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; }}
    </style>
  </head>
  <body>
    <h1>RareSynth App</h1>
    <div class="sub">Run synthetic cohort jobs, inspect runs, and assess biological/quality signals.</div>
    <div class="grid">
      <div class="card">
        <h3>Run Pipeline</h3>
        <label>Config</label>
        <select id="config">
          <option>configs/default_uvm.yaml</option>
          <option>configs/default_lgg.yaml</option>
          <option>configs/cbioportal_ucec_tcga.yaml</option>
        </select>
        <label>Command</label>
        <select id="command">
          <option>all</option><option>download</option><option>preprocess</option>
          <option>train</option><option>validate</option><option>benchmark-models</option><option>ingest-cbioportal</option>
        </select>
        <label>Run ID (optional)</label>
        <input id="runId" placeholder="optional-custom-run-id"/>
        <button onclick="runJob()">Execute</button>
        <div id="runOut" class="mono"></div>
      </div>
      <div class="card">
        <h3>Run Metrics Summary</h3>
        <label>Run ID</label>
        <input id="summaryRunId" placeholder="paste run id"/>
        <button onclick="loadSummary()">Load Summary</button>
        <div id="summaryOut" class="mono"></div>
      </div>
      <div class="card">
        <h3>Compare Runs</h3>
        <label>Run A</label><input id="runA" placeholder="run id A"/>
        <label>Run B</label><input id="runB" placeholder="run id B"/>
        <button onclick="compareRuns()">Compare</button>
        <div id="cmpOut" class="mono"></div>
      </div>
      <div class="card">
        <h3>Recent Runs</h3>
        <button onclick="loadRuns()">Refresh Runs</button>
        <pre id="runsOut"></pre>
      </div>
    </div>
    <script>
      const ROOT = {json.dumps(root)};
      async function runJob() {{
        const body = {{
          config: document.getElementById('config').value,
          root: ROOT,
          command: document.getElementById('command').value,
        }};
        const runId = document.getElementById('runId').value.trim();
        if (runId) body.run_id = runId;
        const r = await fetch('/generate', {{
          method: 'POST',
          headers: {{'Content-Type':'application/json'}},
          body: JSON.stringify(body),
        }});
        const j = await r.json();
        document.getElementById('runOut').textContent = JSON.stringify(j, null, 2);
      }}
      async function loadRuns() {{
        const r = await fetch('/runs?root=' + encodeURIComponent(ROOT));
        const j = await r.json();
        document.getElementById('runsOut').textContent = JSON.stringify(j.rows, null, 2);
      }}
      async function compareRuns() {{
        const a = document.getElementById('runA').value.trim();
        const b = document.getElementById('runB').value.trim();
        let url = '/compare?root=' + encodeURIComponent(ROOT);
        if (a) url += '&run_a=' + encodeURIComponent(a);
        if (b) url += '&run_b=' + encodeURIComponent(b);
        const r = await fetch(url);
        const j = await r.json();
        document.getElementById('cmpOut').textContent = JSON.stringify(j, null, 2);
      }}
      async function loadSummary() {{
        const runId = document.getElementById('summaryRunId').value.trim();
        const r = await fetch('/run-summary?root=' + encodeURIComponent(ROOT) + '&run_id=' + encodeURIComponent(runId));
        const j = await r.json();
        document.getElementById('summaryOut').textContent = JSON.stringify(j, null, 2);
      }}
      loadRuns();
    </script>
  </body>
</html>
"""


@app.post("/generate")
def generate(req: GenerateRequest) -> dict:
    paths = build_paths(Path(req.root))
    config = load_config(req.config)
    run_id = req.run_id or create_run_id(config.cohort_name)

    if req.command not in {"download", "preprocess", "train", "validate", "all", "benchmark-models"}:
        raise HTTPException(
            status_code=400,
            detail="command must be one of download/preprocess/train/validate/all/benchmark-models",
        )

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

    if req.command == "benchmark-models":
        manifest = run_model_benchmark(paths, config, run_id=run_id)
        append_registry_row(
            paths.root,
            run_id,
            config.cohort_name,
            req.config,
            "benchmark-models",
            "success",
            notes=f"manifest={manifest}",
        )
        return {"run_id": run_id, "status": "benchmark_complete", "manifest_path": str(manifest)}

    metrics, model_path, metrics_path = run_all(paths, config, run_id=run_id)
    synthetic_path = (
        paths.root
        / "results"
        / "runs"
        / run_id
        / "synthetic"
        / f"synthetic_{config.train_model}_{config.train_epochs}ep.parquet"
    )
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
