from __future__ import annotations

from pathlib import Path

import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer


def _gene_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if str(c).startswith("ENSG")]


def prepare_training_data(combined_path: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    df = pd.read_parquet(combined_path)

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
    categorical_cols = [c for c in clinical_cols if c != "age_at_diagnosis"]

    if "age_at_diagnosis" in df.columns:
        df["age_at_diagnosis"] = df["age_at_diagnosis"].fillna(df["age_at_diagnosis"].median())

    for col in categorical_cols:
        df[col] = df[col].fillna("unknown").astype(str)

    train_cols = [c for c in numeric_cols + categorical_cols if c in df.columns]
    return df[train_cols].copy(), numeric_cols, categorical_cols


def build_metadata(train_df: pd.DataFrame, categorical_cols: list[str]) -> SingleTableMetadata:
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(train_df)
    for col in categorical_cols:
        if col in train_df.columns:
            metadata.update_column(column_name=col, sdtype="categorical")
    return metadata


def train_ctgan(train_df: pd.DataFrame, metadata: SingleTableMetadata, epochs: int) -> CTGANSynthesizer:
    n = len(train_df)
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

    try:
        synth = CTGANSynthesizer(metadata, **kwargs, pac=1)
    except TypeError:
        synth = CTGANSynthesizer(metadata, **kwargs)

    synth.fit(train_df)
    return synth


def run_train(
    combined_path: Path,
    synthetic_dir: Path,
    epochs: int,
    synthetic_multiplier: int,
    synthetic_min_rows: int,
) -> tuple[Path, Path]:
    synthetic_dir.mkdir(parents=True, exist_ok=True)

    train_df, _num, cat_cols = prepare_training_data(combined_path)
    metadata = build_metadata(train_df, cat_cols)
    synth = train_ctgan(train_df, metadata, epochs=epochs)

    model_path = synthetic_dir / f"model_{epochs}ep.pkl"
    synth.save(str(model_path))

    n_rows = max(synthetic_min_rows, len(train_df) * synthetic_multiplier)
    synthetic = synth.sample(num_rows=n_rows)
    synth_path = synthetic_dir / f"synthetic_{epochs}ep.parquet"
    synthetic.to_parquet(synth_path)

    return model_path, synth_path
