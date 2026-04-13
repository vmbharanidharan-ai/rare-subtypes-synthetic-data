"""
Validate synthetic data: UMAP overlap, gene-level KS stats, TSTR utility, privacy proxy.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

try:
    import umap
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install umap-learn: pip install umap-learn") from exc

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results" / "synthetic"
FIG_DIR = ROOT / "results" / "figures"


def _gene_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if str(c).startswith("ENSG")]


def load_data(synthetic_name: str = "synthetic_100ep.parquet"):
    real = pd.read_parquet(PROC_DIR / "combined.parquet")
    synthetic = pd.read_parquet(RESULTS_DIR / synthetic_name)
    return real, synthetic, _gene_columns(real)


def plot_umap(real: pd.DataFrame, synthetic: pd.DataFrame, gene_cols: list[str]) -> None:
    shared = [g for g in gene_cols if g in synthetic.columns]
    if len(shared) < 10:
        print("UMAP skipped: not enough overlapping gene columns")
        return
    real_genes = real[shared].values.astype(float)
    synth_genes = synthetic[shared].values.astype(float)

    pca = PCA(n_components=min(50, real_genes.shape[1]), random_state=42)
    real_pca = pca.fit_transform(real_genes)
    synth_pca = pca.transform(synth_genes)

    combined_pca = np.vstack([real_pca, synth_pca])
    labels = np.array(["Real"] * len(real_pca) + ["Synthetic"] * len(synth_pca))

    n_neighbors = max(5, min(15, len(real_pca) - 1))
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors)
    embedding = reducer.fit_transform(combined_pca)

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"Real": "#1D9E75", "Synthetic": "#D85A30"}
    for label, color in colors.items():
        mask = labels == label
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=color,
            label=label,
            alpha=0.7,
            s=50 if label == "Real" else 18,
        )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("UMAP: real vs synthetic (PCA 50d input)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_DIR / "umap_real_vs_synthetic.png", dpi=150)
    plt.close()
    print("Saved UMAP")


def compute_ks_stats(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    gene_cols: list[str],
    n_genes: int = 500,
) -> pd.DataFrame:
    shared = [g for g in gene_cols if g in synthetic.columns]
    rng = np.random.default_rng(42)
    sample_genes = list(
        shared
        if len(shared) <= n_genes
        else rng.choice(shared, size=n_genes, replace=False)
    )

    rows = []
    for gene in sample_genes:
        r = pd.to_numeric(real[gene], errors="coerce").dropna()
        s = pd.to_numeric(synthetic[gene], errors="coerce").dropna()
        if len(r) < 3 or len(s) < 3:
            continue
        stat, pval = ks_2samp(r, s)
        rows.append({"gene": gene, "ks_stat": stat, "p_value": pval})

    df = pd.DataFrame(rows)
    if df.empty:
        print("KS stats: no genes compared")
        return df

    pct_good = (df["ks_stat"] < 0.2).mean() * 100
    print(f"KS: {pct_good:.1f}% of genes with KS < 0.2")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["ks_stat"], bins=40, color="#378ADD", edgecolor="white", linewidth=0.5)
    ax.axvline(0.2, color="#D85A30", linestyle="--", label="KS = 0.2")
    ax.set_xlabel("KS statistic (lower = closer distributions)")
    ax.set_ylabel("Genes")
    ax.set_title("Gene-level KS: real vs synthetic")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_DIR / "ks_distribution.png", dpi=150)
    plt.close()
    print("Saved KS histogram")
    return df


def _align_synthetic_labels(synth_vital: pd.Series, le: LabelEncoder) -> np.ndarray:
    """Map synthetic vital_status strings to the same integer codes as `le`."""
    classes = list(le.classes_)
    default = classes.index("unknown") if "unknown" in classes else 0
    out = []
    for v in synth_vital.astype(str):
        out.append(le.transform([v])[0] if v in classes else default)
    return np.array(out, dtype=int)


def tstr_utility_test(real: pd.DataFrame, synthetic: pd.DataFrame, gene_cols: list[str]) -> None:
    if "vital_status" not in real.columns or "vital_status" not in synthetic.columns:
        print("TSTR skipped: vital_status missing")
        return

    shared = [g for g in gene_cols if g in synthetic.columns][: min(200, len(gene_cols))]
    if len(shared) < 5:
        print("TSTR skipped: too few genes")
        return

    real_x = real[shared].fillna(0).values.astype(float)
    real_labels = real["vital_status"].fillna("unknown").astype(str)
    le = LabelEncoder()
    real_y = le.fit_transform(real_labels)
    if len(np.unique(real_y)) < 2:
        print("TSTR skipped: vital_status has a single class in real data")
        return

    synth_x = synthetic[shared].fillna(0).values.astype(float)
    synth_y = _align_synthetic_labels(synthetic["vital_status"], le)

    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    n_splits = min(5, len(real_y) // 2)
    n_splits = max(2, n_splits)
    try:
        trtr_scores = cross_val_score(
            clf,
            real_x,
            real_y,
            cv=n_splits,
            scoring="roc_auc",
        )
    except ValueError as e:
        print(f"TSTR skipped (CV): {e}")
        return

    clf.fit(synth_x, synth_y)
    if clf.predict_proba(real_x).shape[1] < 2:
        print("TSTR skipped: classifier returned single-class probabilities")
        return
    tstr_auc = roc_auc_score(real_y, clf.predict_proba(real_x)[:, 1])

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(
        ["TRTR (baseline)", "TSTR (synth→real)"],
        [float(np.mean(trtr_scores)), float(tstr_auc)],
        color=["#1D9E75", "#D85A30"],
        width=0.45,
    )
    ax.errorbar(
        0,
        float(np.mean(trtr_scores)),
        yerr=float(np.std(trtr_scores)),
        fmt="none",
        color="black",
        capsize=5,
    )
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Utility (TSTR vs TRTR)")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "tstr_utility.png", dpi=150)
    plt.close()

    print(f"TRTR mean AUC: {np.mean(trtr_scores):.3f} ± {np.std(trtr_scores):.3f}")
    print(f"TSTR AUC: {tstr_auc:.3f}")


def privacy_membership_proxy(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    gene_cols: list[str],
) -> None:
    shared = [g for g in gene_cols if g in synthetic.columns][: min(200, len(gene_cols))]
    if len(shared) < 5:
        print("Privacy proxy skipped")
        return

    real_x = real[shared].fillna(0).values.astype(float)
    synth_x = synthetic[shared].fillna(0).values.astype(float)
    x = np.vstack([real_x, synth_x])
    y = np.array([1] * len(real_x) + [0] * len(synth_x))

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    skf = StratifiedKFold(n_splits=min(5, len(y) // 2), shuffle=True, random_state=42)
    aucs = []
    for train_idx, test_idx in skf.split(x, y):
        clf.fit(x[train_idx], y[train_idx])
        prob = clf.predict_proba(x[test_idx])[:, 1]
        aucs.append(roc_auc_score(y[test_idx], prob))

    mean_auc = float(np.mean(aucs))
    print(f"\nPrivacy proxy — real vs synthetic classifier AUC: {mean_auc:.3f}")
    if mean_auc < 0.70:
        print("  Interpretation: synthetic not trivially separable from real (good).")
    elif mean_auc < 0.85:
        print("  Interpretation: moderate separability — tune generative model / noise.")
    else:
        print("  Interpretation: high separability — risk of memorization / overfit.")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading data...")
    real, synthetic, gene_cols = load_data()

    print("\n=== UMAP ===")
    plot_umap(real, synthetic, gene_cols)

    print("\n=== KS ===")
    ks_df = compute_ks_stats(real, synthetic, gene_cols)
    if not ks_df.empty:
        ks_df.to_csv(FIG_DIR / "ks_results.csv", index=False)

    print("\n=== TSTR ===")
    tstr_utility_test(real, synthetic, gene_cols)

    print("\n=== Privacy proxy ===")
    privacy_membership_proxy(real, synthetic, gene_cols)

    print(f"\nFigures in {FIG_DIR}")


if __name__ == "__main__":
    main()
