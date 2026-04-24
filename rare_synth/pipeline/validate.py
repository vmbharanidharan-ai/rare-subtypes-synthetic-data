from __future__ import annotations

import json
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

import umap


def _gene_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if str(c).startswith("ENSG")]


def plot_umap(real: pd.DataFrame, synthetic: pd.DataFrame, gene_cols: list[str], out_path: Path) -> None:
    shared = [g for g in gene_cols if g in synthetic.columns]
    if len(shared) < 10:
        return

    real_genes = real[shared].values.astype(float)
    synth_genes = synthetic[shared].values.astype(float)

    pca = PCA(n_components=min(50, real_genes.shape[1]), random_state=42)
    real_pca = pca.fit_transform(real_genes)
    synth_pca = pca.transform(synth_genes)

    embed = umap.UMAP(n_components=2, random_state=42, n_neighbors=max(5, min(15, len(real_pca) - 1)))
    embedding = embed.fit_transform(np.vstack([real_pca, synth_pca]))
    labels = np.array(["Real"] * len(real_pca) + ["Synthetic"] * len(synth_pca))

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"Real": "#1D9E75", "Synthetic": "#D85A30"}
    for label, color in colors.items():
        mask = labels == label
        ax.scatter(embedding[mask, 0], embedding[mask, 1], c=color, label=label, alpha=0.7, s=18)
    ax.set_title("UMAP: real vs synthetic")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def compute_ks_stats(real: pd.DataFrame, synthetic: pd.DataFrame, gene_cols: list[str], n_genes: int) -> pd.DataFrame:
    shared = [g for g in gene_cols if g in synthetic.columns]
    rng = np.random.default_rng(42)
    sample_genes = shared if len(shared) <= n_genes else list(rng.choice(shared, size=n_genes, replace=False))

    rows = []
    for gene in sample_genes:
        rv = pd.to_numeric(real[gene], errors="coerce").dropna()
        sv = pd.to_numeric(synthetic[gene], errors="coerce").dropna()
        if len(rv) < 3 or len(sv) < 3:
            continue
        stat, pval = ks_2samp(rv, sv)
        rows.append({"gene": gene, "ks_stat": float(stat), "p_value": float(pval)})

    return pd.DataFrame(rows)


def _align_synthetic_labels(synth_vital: pd.Series, le: LabelEncoder) -> np.ndarray:
    classes = list(le.classes_)
    default = classes.index("unknown") if "unknown" in classes else 0
    out = []
    for val in synth_vital.astype(str):
        out.append(le.transform([val])[0] if val in classes else default)
    return np.array(out, dtype=int)


def tstr_utility(real: pd.DataFrame, synthetic: pd.DataFrame, gene_cols: list[str], tstr_max_genes: int) -> dict[str, float | None]:
    if "vital_status" not in real.columns or "vital_status" not in synthetic.columns:
        return {"trtr_auc": None, "tstr_auc": None}

    shared = [g for g in gene_cols if g in synthetic.columns][: min(tstr_max_genes, len(gene_cols))]
    if len(shared) < 5:
        return {"trtr_auc": None, "tstr_auc": None}

    real_x = real[shared].fillna(0).values.astype(float)
    real_y_raw = real["vital_status"].fillna("unknown").astype(str)
    le = LabelEncoder()
    real_y = le.fit_transform(real_y_raw)
    if len(np.unique(real_y)) < 2:
        return {"trtr_auc": None, "tstr_auc": None}

    synth_x = synthetic[shared].fillna(0).values.astype(float)
    synth_y = _align_synthetic_labels(synthetic["vital_status"], le)

    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    n_splits = max(2, min(5, len(real_y) // 2))

    try:
        trtr_scores = cross_val_score(clf, real_x, real_y, cv=n_splits, scoring="roc_auc")
    except ValueError:
        return {"trtr_auc": None, "tstr_auc": None}

    clf.fit(synth_x, synth_y)
    prob = clf.predict_proba(real_x)
    if prob.shape[1] < 2:
        return {"trtr_auc": float(np.mean(trtr_scores)), "tstr_auc": None}

    return {"trtr_auc": float(np.mean(trtr_scores)), "tstr_auc": float(roc_auc_score(real_y, prob[:, 1]))}


def privacy_proxy_auc(real: pd.DataFrame, synthetic: pd.DataFrame, gene_cols: list[str], tstr_max_genes: int) -> float | None:
    shared = [g for g in gene_cols if g in synthetic.columns][: min(tstr_max_genes, len(gene_cols))]
    if len(shared) < 5:
        return None

    real_x = real[shared].fillna(0).values.astype(float)
    synth_x = synthetic[shared].fillna(0).values.astype(float)
    x = np.vstack([real_x, synth_x])
    y = np.array([1] * len(real_x) + [0] * len(synth_x))

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    n_splits = max(2, min(5, len(y) // 2))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    aucs = []
    for train_idx, test_idx in skf.split(x, y):
        clf.fit(x[train_idx], y[train_idx])
        prob = clf.predict_proba(x[test_idx])[:, 1]
        aucs.append(roc_auc_score(y[test_idx], prob))

    return float(np.mean(aucs))


def nearest_neighbor_leakage(
    real: pd.DataFrame, synthetic: pd.DataFrame, gene_cols: list[str], tstr_max_genes: int
) -> float | None:
    """
    Proxy leakage metric: fraction of synthetic samples whose nearest real neighbor
    is *very* close in standardized feature space. Higher => more memorization risk.
    """
    shared = [g for g in gene_cols if g in synthetic.columns][: min(tstr_max_genes, len(gene_cols))]
    if len(shared) < 5:
        return None

    real_x = real[shared].fillna(0).values.astype(float)
    synth_x = synthetic[shared].fillna(0).values.astype(float)

    # Standardize using real distribution only.
    mu = real_x.mean(axis=0, keepdims=True)
    sigma = real_x.std(axis=0, keepdims=True)
    sigma[sigma == 0] = 1.0
    real_z = (real_x - mu) / sigma
    synth_z = (synth_x - mu) / sigma

    # Compute nearest real distance for each synthetic row.
    # Chunk to avoid blowing memory on larger cohorts.
    chunk = 128
    min_dists = []
    for start in range(0, len(synth_z), chunk):
        end = min(start + chunk, len(synth_z))
        block = synth_z[start:end]
        d2 = ((block[:, None, :] - real_z[None, :, :]) ** 2).sum(axis=2)
        min_dists.extend(np.sqrt(d2.min(axis=1)))
    min_dists = np.array(min_dists, dtype=float)
    if len(min_dists) == 0:
        return None

    # "Too close" threshold tuned for z-scored high-d data:
    # we report fraction below 1.0 Euclidean distance.
    return float((min_dists < 1.0).mean())


def write_validation_report(metrics: dict, report_json: Path, report_md: Path) -> None:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    lines = [
        "# RareSynth Validation Report",
        "",
        f"- Samples (real): {metrics.get('n_real')}",
        f"- Samples (synthetic): {metrics.get('n_synthetic')}",
        f"- KS genes evaluated: {metrics.get('ks_genes_evaluated')}",
        f"- KS<0.2 (%): {metrics.get('ks_pct_below_0_2')}",
        f"- TRTR AUC: {metrics.get('trtr_auc')}",
        f"- TSTR AUC: {metrics.get('tstr_auc')}",
        f"- Privacy proxy AUC (real-vs-synth): {metrics.get('privacy_proxy_auc')}",
        f"- NN leakage ratio (<1.0 z-dist): {metrics.get('nn_leakage_ratio')}",
        "",
        "Interpretation:",
        "- Higher TSTR relative to TRTR indicates better synthetic utility.",
        "- Privacy proxy AUC closer to 0.5 indicates synthetic data less separable from real.",
        "- Lower NN leakage ratio indicates lower memorization risk.",
    ]
    report_md.write_text("\n".join(lines), encoding="utf-8")


def run_validate(
    combined_path: Path,
    synthetic_path: Path,
    figures_dir: Path,
    reports_dir: Path,
    sample_ks_genes: int,
    tstr_max_genes: int,
) -> dict:
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    real = pd.read_parquet(combined_path)
    synthetic = pd.read_parquet(synthetic_path)
    gene_cols = _gene_columns(real)

    plot_umap(real, synthetic, gene_cols, figures_dir / "umap_real_vs_synthetic.png")

    ks_df = compute_ks_stats(real, synthetic, gene_cols, n_genes=sample_ks_genes)
    if not ks_df.empty:
        ks_df.to_csv(figures_dir / "ks_results.csv", index=False)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(ks_df["ks_stat"], bins=40, color="#378ADD", edgecolor="white", linewidth=0.5)
        ax.axvline(0.2, color="#D85A30", linestyle="--", label="KS = 0.2")
        ax.set_title("Gene-level KS: real vs synthetic")
        ax.set_xlabel("KS statistic")
        ax.set_ylabel("Genes")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / "ks_distribution.png", dpi=150)
        plt.close(fig)

    utility = tstr_utility(real, synthetic, gene_cols, tstr_max_genes=tstr_max_genes)
    privacy_auc = privacy_proxy_auc(real, synthetic, gene_cols, tstr_max_genes=tstr_max_genes)
    nn_leakage = nearest_neighbor_leakage(real, synthetic, gene_cols, tstr_max_genes=tstr_max_genes)

    metrics = {
        "n_real": int(len(real)),
        "n_synthetic": int(len(synthetic)),
        "ks_genes_evaluated": int(len(ks_df)) if not ks_df.empty else 0,
        "ks_pct_below_0_2": float((ks_df["ks_stat"] < 0.2).mean() * 100.0) if not ks_df.empty else None,
        "trtr_auc": utility.get("trtr_auc"),
        "tstr_auc": utility.get("tstr_auc"),
        "privacy_proxy_auc": privacy_auc,
        "nn_leakage_ratio": nn_leakage,
    }

    write_validation_report(
        metrics,
        report_json=reports_dir / "validation_metrics.json",
        report_md=reports_dir / "validation_report.md",
    )
    return metrics
