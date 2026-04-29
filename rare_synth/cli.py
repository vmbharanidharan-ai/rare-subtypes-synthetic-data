from __future__ import annotations

import argparse
from pathlib import Path

from rare_synth.config import load_config
from rare_synth.paths import build_paths
from rare_synth.pipeline.compare import compare_runs
from rare_synth.pipeline.orchestrator import (
    create_run_id,
    run_all,
    run_cbioportal_ingest,
    run_download,
    run_model_benchmark,
    run_preprocessing,
    run_training,
    run_validation,
)
from rare_synth.pipeline.registry import append_registry_row
from rare_synth.pipeline.registry import load_registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RareSynth pipeline runner")
    parser.add_argument(
        "command",
        choices=[
            "download",
            "preprocess",
            "train",
            "validate",
            "all",
            "ingest-cbioportal",
            "compare-runs",
            "benchmark-models",
            "serve-api",
            "list-runs",
        ],
        help="Pipeline stage to run",
    )
    parser.add_argument("--config", default="configs/default_uvm.yaml", help="Path to YAML config")
    parser.add_argument("--root", default=".", help="Project root directory (default: current directory)")
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run id for versioned artifacts. Defaults to <cohort>-<utc-timestamp>",
    )
    parser.add_argument("--run-a", default=None, help="First run id for compare-runs")
    parser.add_argument("--run-b", default=None, help="Second run id for compare-runs")
    parser.add_argument("--host", default="127.0.0.1", help="Host for serve-api")
    parser.add_argument("--port", default=8000, type=int, help="Port for serve-api")
    return parser


def _register(
    root: Path,
    run_id: str,
    cohort_name: str,
    config_path: str,
    stage: str,
    status: str,
    model_path: str | None = None,
    synthetic_path: str | None = None,
    metrics_path: str | None = None,
    notes: str | None = None,
) -> None:
    append_registry_row(
        root=root,
        run_id=run_id,
        cohort_name=cohort_name,
        config_path=config_path,
        stage=stage,
        status=status,
        model_path=model_path,
        synthetic_path=synthetic_path,
        metrics_path=metrics_path,
        notes=notes,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    paths = build_paths(Path(args.root))

    if args.command == "serve-api":
        import uvicorn

        uvicorn.run("rare_synth.api:app", host=args.host, port=args.port, reload=False)
        return

    if args.command == "list-runs":
        df = load_registry(paths.root)
        if df.empty:
            print("No runs found in results/runs/index.csv")
            return
        print(df.tail(30).to_string(index=False))
        return

    if args.command == "compare-runs":
        result = compare_runs(paths.root, run_a=args.run_a, run_b=args.run_b)
        print(result)
        return

    config = load_config(args.config)
    run_id = args.run_id or create_run_id(config.cohort_name)

    if args.command == "download":
        run_download(paths, config)
        return

    if args.command == "ingest-cbioportal":
        out_dir = run_cbioportal_ingest(paths, config)
        _register(
            root=paths.root,
            run_id=run_id,
            cohort_name=config.cohort_name,
            config_path=args.config,
            stage="ingest-cbioportal",
            status="success",
            notes=f"snapshot={out_dir}",
        )
        print(f"cBioPortal snapshot -> {out_dir}")
        return

    if args.command == "preprocess":
        out = run_preprocessing(paths, config)
        _register(
            root=paths.root,
            run_id=run_id,
            cohort_name=config.cohort_name,
            config_path=args.config,
            stage="preprocess",
            status="success",
            notes=f"combined={out}",
        )
        print(f"Preprocessed dataset -> {out}")
        return

    if args.command == "train":
        model_path, synth_path = run_training(paths, config, run_id=run_id)
        _register(
            root=paths.root,
            run_id=run_id,
            cohort_name=config.cohort_name,
            config_path=args.config,
            stage="train",
            status="success",
            model_path=str(model_path),
            synthetic_path=str(synth_path),
        )
        print(f"Run id -> {run_id}")
        print(f"Model -> {model_path}")
        print(f"Synthetic data -> {synth_path}")
        return

    if args.command == "benchmark-models":
        manifest = run_model_benchmark(paths, config, run_id=run_id)
        _register(
            root=paths.root,
            run_id=run_id,
            cohort_name=config.cohort_name,
            config_path=args.config,
            stage="benchmark-models",
            status="success",
            notes=f"manifest={manifest}",
        )
        print(f"Run id -> {run_id}")
        print(f"Benchmark manifest -> {manifest}")
        return

    if args.command == "validate":
        metrics, metrics_path = run_validation(paths, config, run_id=run_id)
        _register(
            root=paths.root,
            run_id=run_id,
            cohort_name=config.cohort_name,
            config_path=args.config,
            stage="validate",
            status="success",
            metrics_path=str(metrics_path),
        )
        print(f"Run id -> {run_id}")
        print(metrics)
        return

    metrics, model_path, metrics_path = run_all(paths, config, run_id=run_id)
    synthetic_path = (
        paths.root
        / "results"
        / "runs"
        / run_id
        / "synthetic"
        / f"synthetic_{config.train_model}_{config.train_epochs}ep.parquet"
    )
    _register(
        root=paths.root,
        run_id=run_id,
        cohort_name=config.cohort_name,
        config_path=args.config,
        stage="all",
        status="success",
        model_path=str(model_path),
        synthetic_path=str(synthetic_path),
        metrics_path=str(metrics_path),
    )
    print(f"Run id -> {run_id}")
    print(metrics)


if __name__ == "__main__":
    main()
