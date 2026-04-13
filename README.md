# rare-synth (MVP)

End-to-end demo: **open TCGA-UVM RNA-seq** → preprocess → **CTGAN (SDV)** → synthetic cohort → **UMAP / KS / TSTR / privacy proxy** figures.

## Prerequisites (Mac)

- Python **3.11+** (e.g. `brew install python@3.11`)
- **Cursor**: open this folder (`File → Open Folder…`)

## Setup

```bash
cd /Users/vedhabharanidharan/Projects/rare-synth
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## API keys

- **GDC / TCGA open access**: no key (this MVP uses only open STAR count files).
- **cBioPortal** (optional later): public API, no key.
- **Controlled-access TCGA** (optional, not needed here): NIH **eRA Commons** + `gdc-client` + dbGaP authorization.

## Run pipeline

```bash
source venv/bin/activate
python src/download.py      # ~200–500 MB download
python src/preprocess.py
python src/train.py         # CTGAN; 100 epochs ≈ 15–40 min on Apple Silicon
python src/validate.py
```

Outputs:

- `data/processed/combined.parquet` — real table used for training  
- `results/synthetic/synthetic_100ep.parquet` — synthetic patients  
- `results/figures/*.png` — UMAP, KS histogram, TSTR bar chart  

## Cursor workflow

1. Open the **Terminal** in Cursor (`Ctrl+` `).
2. Run commands above; paste errors into chat with **Cmd+L** / Agent for fixes.
3. Tune CTGAN: edit `epochs`, `generator_dim`, or `batch_size` in `src/train.py`.

## Datasets (this MVP)

| Source | Cohort | What you pull |
|--------|--------|----------------|
| [GDC](https://portal.gdc.cancer.gov/) | **TCGA-UVM** | RNA-seq **Gene Expression Quantification**, workflow **STAR - Counts**, access **open** |

## Optional next steps

- Add **cBioPortal** cohorts for more UVM samples (REST API).
- Add **synthcity** for formal privacy metrics.
- Swap CTGAN for **scVI** / **TVAE** paths for single-cell or different priors.
