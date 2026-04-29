# rare-synth (product-ready demo v1.3)

Config-driven synthetic cohort generation for rare oncology with versioned runs, run registry, run comparison, and API access.

## Purpose

`rare-synth` exists to help researchers and AI teams work on **rare cancer subtypes** (and eventually broader rare diseases) where real patient cohorts are too small for robust model development.

The platform:
- ingests public cohort data (currently GDC and cBioPortal snapshots),
- generates synthetic cohorts with CTGAN,
- evaluates whether synthetic data is useful and safe enough for downstream experimentation,
- and tracks every run so results are reproducible and comparable.

This is a research/development platform, not a clinical decision tool.

## Summary At A Glance

- **Problem:** rare subtype datasets are sparse and fragmented.
- **Core workflow:** `download -> preprocess -> train -> validate`.
- **Primary output:** synthetic cohorts plus validation reports and figures.
- **Main interfaces:** CLI, API (`/docs`), and product UI (`/app`).
- **Main value:** faster iteration on rare-subtype modeling with explicit quality signals.

## Who This Is For

- Computational oncology researchers
- Bioinformatics / ML teams building rare-disease models

## What v1.3 adds

- **Run registry**: `results/runs/index.csv`
- **Run comparison command**: compare metric deltas between runs
- **FastAPI service**: run generation and inspect registry over HTTP
- **Dashboard**: `/dashboard` run monitor page
- **Resilient GDC downloads**: retry + tar integrity validation
- **Extra privacy metric**: nearest-neighbor leakage ratio

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

### Architecture diagram

```text
                  +-----------------------------+
                  |         User / Team         |
                  |   CLI, API (/generate), UI |
                  +-------------+---------------+
                                |
                                v
                    +-----------+-----------+
                    |   Orchestrator/CLI    |
                    |  (download->validate) |
                    +-----------+-----------+
                                |
            +-------------------+-------------------+
            |                                       |
            v                                       v
 +----------+-----------+                 +---------+----------+
 | Data Ingestion       |                 | Run Registry       |
 | GDC / cBioPortal     |                 | results/runs/index |
 +----------+-----------+                 +---------+----------+
            |                                       |
            v                                       |
 +----------+-----------+                           |
 | Preprocess           |                           |
 | expression + clinical|                           |
 +----------+-----------+                           |
            |                                       |
            v                                       |
 +----------+-----------+                           |
 | Train (CTGAN)        |                           |
 | synthetic cohort     |                           |
 +----------+-----------+                           |
            |                                       |
            v                                       |
 +----------+-----------+                           |
 | Validate             |<--------------------------+
 | fidelity/utility/    |
 | privacy/clinical     |
 +----------+-----------+
            |
            v
 +----------+-----------+
 | Artifacts            |
 | synthetic, figures,  |
 | reports, summaries   |
 +----------------------+
```

## Setup

```bash
# from repository root
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run with Docker

Build image:

```bash
docker build -t rare-synth:latest .
```

Run API:

```bash
docker run --rm -p 8000:8000 \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/results:/app/results" \
  rare-synth:latest
```

Or with Docker Compose:

```bash
docker compose up --build
```

Stop:

```bash
docker compose down
```

Then open:
- `http://127.0.0.1:8000/app?root=.`
- `http://127.0.0.1:8000/docs`

Notes:
- The mounted `data/` and `results/` directories persist outputs on your host.
- If port `8000` is already used, map another port (for example `-p 8001:8000`).

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

### List recent runs (operator view)

```bash
python -m rare_synth.cli list-runs --root .
```

### Serve API

```bash
python -m rare_synth.cli serve-api --host 127.0.0.1 --port 8000
```

## API endpoints

- `GET /health`
- `GET /runs?root=.`
- `GET /compare?root=.&run_a=...&run_b=...`
- `GET /run-summary?root=.&run_id=...`
- `GET /dashboard?root=.`
- `GET /app?root=.`
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

## Product UI (recommended)

Run:

```bash
python3 -m rare_synth.cli serve-api --host 127.0.0.1 --port 8000
```

Open:

- `http://127.0.0.1:8000/app?root=.`

The app page supports:
- run execution (`download/preprocess/train/validate/all`)
- run comparison
- per-run metric summary with quality grade
- recent run registry inspection

## Biological usefulness signals

Validation now includes:
- distribution fidelity (`ks_pct_below_0_2`)
- utility proxies (`trtr_auc`, `tstr_auc` when labels are usable)
- privacy/separability (`privacy_proxy_auc`)
- nearest-neighbor leakage risk (`nn_leakage_ratio`)
- clinical consistency proxies:
  - `age_mean_abs_diff`
  - `vital_status_dist_l1`

## Troubleshooting

- `EOFError: Compressed file ended...` during GDC extraction:
  - Usually a truncated download.
  - Re-run download; the downloader now retries and validates archive integrity.
- `trtr_auc` / `tstr_auc` is `None`:
  - Target labels are missing or single-class after filtering.
  - This is expected for some cohorts; use comparison over cohorts/runs.
- `privacy_proxy_auc` near `1.0`:
  - Real and synthetic are highly separable.
  - Indicates poor fidelity/generalization; tune model, features, or training setup.
- `nn_leakage_ratio` high:
  - Synthetic rows are too close to real rows in standardized feature space.
  - Indicates potential memorization risk.

## Backward-compatible wrappers

These still work and call the new CLI:

```bash
python src/download.py
python src/preprocess.py
python src/train.py
python src/validate.py
```

---

## Multimodal Extension Project

The repository also includes a second project scaffold at `multimodal-rare-synth/` for the full multimodal rare-oncology roadmap.

### Scope

`multimodal-rare-synth` is designed for synthetic generation across:
1. imaging (histopathology / MRI / CT)
2. genomic / multi-omics
3. clinical tabular / longitudinal
4. cross-modal integration
5. ontology-informed generation for ultra-rare subtypes

### What this scaffold provides

- modular interfaces for each modality-specific generator
- one orchestrator to run the full multimodal stack
- validation hooks for fidelity / utility / privacy / biological consistency
- API skeleton to trigger generation jobs

### Multimodal architecture

```text
Data Sources -> Harmonization -> Modality Generators -> Cross-Modal Consistency -> Validation -> Artifacts/API

Modalities:
  ImagingGenerator
  GenomicsGenerator
  ClinicalGenerator
  MultiOmicsIntegrator
  OntologyGenerator
```

### Quick start (multimodal scaffold)

```bash
cd multimodal-rare-synth
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m multimodal_rare_synth.api.server
```

Then open `http://127.0.0.1:8010/docs`.

### Next implementation priorities

1. plug in a real imaging model (StyleGAN/latent diffusion)
2. plug in omics model (VAE/Transformer + condition vectors)
3. add graph/ontology encoder for ultra-rare transfer
4. implement cross-modal consistency losses + calibration
5. add clinical expert review pipeline and model cards
