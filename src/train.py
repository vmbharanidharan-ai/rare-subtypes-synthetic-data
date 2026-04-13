"""
Train CTGAN (via SDV) on processed TCGA-UVM table; sample synthetic patients.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results" / "synthetic"


def _gene_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if str(c).startswith("ENSG")]


def prepare_training_data(combined_path: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    df = pd.read_parquet(combined_path)
    print(f"Loaded combined: {df.shape}")

    gene_cols = _gene_columns(df)
    clinical_cols = [
        "age_at_diagnosis",
        "vital_status",
        "tumor_stage",
        "gender",
        "race",
    ]
    clinical_cols = [c for c in clinical_cols if c in df.columns]

    numeric_cols = gene_cols + ["age_at_diagnosis"]
    categorical_cols = [
        c
        for c in clinical_cols
        if c not in ("age_at_diagnosis", "days_to_death", "days_to_last_follow_up")
    ]

    if "age_at_diagnosis" in df.columns:
        med = df["age_at_diagnosis"].median()
        df["age_at_diagnosis"] = df["age_at_diagnosis"].fillna(med)

    for col in categorical_cols:
        df[col] = df[col].fillna("unknown").astype(str)

    train_cols = [c for c in numeric_cols + categorical_cols if c in df.columns]
    train_df = df[train_cols].copy()
    numeric_cols = [c for c in numeric_cols if c in train_df.columns]

    print(
        f"Training columns: {len(numeric_cols)} numeric, "
        f"{len(categorical_cols)} categorical"
    )
    return train_df, numeric_cols, categorical_cols


def build_metadata(train_df: pd.DataFrame, categorical_cols: list[str]) -> SingleTableMetadata:
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(train_df)
    for col in categorical_cols:
        if col in train_df.columns:
            metadata.update_column(column_name=col, sdtype="categorical")
    return metadata


def train_ctgan(
    train_df: pd.DataFrame,
    metadata: SingleTableMetadata,
    epochs: int = 300,
) -> CTGANSynthesizer:
    n = len(train_df)
    # CTGAN requires batch_size divisible by pac (default pac=10 in underlying lib)
    pac = 10
    batch_size = min(500, (n // pac) * pac) or n
    if batch_size < pac:
        batch_size = n

    kwargs = dict(
        epochs=epochs,
        verbose=True,
        generator_dim=(256, 256),
        discriminator_dim=(256, 256),
        batch_size=batch_size,
    )
    # SDV passes through CTGAN-specific args when supported
    try:
        synthesizer = CTGANSynthesizer(metadata, **kwargs, pac=1)
    except TypeError:
        synthesizer = CTGANSynthesizer(metadata, **kwargs)

    print(f"Training CTGAN on {n} rows, batch_size={batch_size}, epochs={epochs}...")
    synthesizer.fit(train_df)
    return synthesizer


def generate_synthetic(synthesizer: CTGANSynthesizer, n_samples: int = 500) -> pd.DataFrame:
    synthetic = synthesizer.sample(num_rows=n_samples)
    print(f"Generated {len(synthetic)} synthetic rows")
    return synthetic


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    combined = PROC_DIR / "combined.parquet"
    if not combined.exists():
        raise SystemExit(f"Missing {combined}. Run preprocess.py first.")

    train_df, _num, cat_cols = prepare_training_data(combined)
    metadata = build_metadata(train_df, cat_cols)

    print("=== Quick run: 100 epochs ===")
    synth = train_ctgan(train_df, metadata, epochs=100)
    synth.save(str(RESULTS_DIR / "model_100ep.pkl"))

    synthetic = generate_synthetic(synth, n_samples=max(200, len(train_df) * 3))
    synthetic.to_parquet(RESULTS_DIR / "synthetic_100ep.parquet")
    print(f"Wrote {RESULTS_DIR / 'synthetic_100ep.parquet'}")

    # Uncomment for longer training:
    # synth300 = train_ctgan(train_df, metadata, epochs=300)
    # synth300.save(str(RESULTS_DIR / "model_300ep.pkl"))
    # generate_synthetic(synth300, n_samples=500).to_parquet(
    #     RESULTS_DIR / "synthetic_300ep.parquet"
    # )


if __name__ == "__main__":
    main()
