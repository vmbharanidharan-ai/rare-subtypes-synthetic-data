# rare-synth (platform-style MVP v1.2)

Config-driven synthetic cohort generation for rare oncology with versioned runs, run registry, run comparison, and API access.

## What v1.2 adds

- **Run registry**: `results/runs/index.csv`
- **Run comparison command**: compare metric deltas between runs
- **FastAPI service**: run generation and inspect registry over HTTP

## Core architecture

- `rare_synth/data/gdc.py` - GDC cohort download
- `rare_synth/data/cbioportal.py` - cBioPortal snapshot ingestion
- `rare_synth/pipeline/preprocess.py` - harmonization + HVG selection
- `rare_synth/pipeline/train.py` - CTGAN training + synthetic cohort sampling
- `rare_synth/pipeline/validate.py` - fidelity/utility/privacy metrics + reports
- `rare_synth/pipeline/registry.py` - append/load run registry index
- `rare_synth/pipeline/compare.py` - compare validation metrics across runs
- `rare_synth/api.py` - FastAPI app
- `rare_synth/cli.py` - command runner

## Setup

```bash
cd /Users/vedhabharanidharan/Projects/rare-synth
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## CLI commands

### Full pipeline

```bash
python -m rare_synth.cli all --config configs/default_uvm.yaml --root .
```

### Stage-by-stage with explicit run id

```bash
python -m rare_synth.cli train --config configs/default_uvm.yaml --root . --run-id uvm-test
python -m rare_synth.cli validate --config configs/default_uvm.yaml --root . --run-id uvm-test
```

### Compare runs

```bash
python -m rare_synth.cli compare-runs --root . --run-a run-id-1 --run-b run-id-2
```

If `--run-a` / `--run-b` are omitted, the command compares the latest two run ids in the registry.

### Serve API

```bash
python -m rare_synth.cli serve-api --host 127.0.0.1 --port 8000
```

## API endpoints

- `GET /health`
- `GET /runs?root=.`
- `GET /compare?root=.&run_a=...&run_b=...`
- `POST /generate`

Example payload:

```json
{
  "config": "configs/default_uvm.yaml",
  "root": ".",
  "command": "all",
  "run_id": "optional-custom-id"
}
```

## Configs included

- `configs/default_uvm.yaml`
- `configs/default_lgg.yaml`
- `configs/cbioportal_ucec_tcga.yaml`

## Artifacts

Per run id, artifacts are stored under:

- `results/runs/<run_id>/synthetic/`
- `results/runs/<run_id>/figures/`
- `results/runs/<run_id>/reports/`

Run index across all runs:

- `results/runs/index.csv`

## Backward-compatible wrappers

These still work and call the new CLI:

```bash
python src/download.py
python src/preprocess.py
python src/train.py
python src/validate.py
```
