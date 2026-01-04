"""
RQ1 - Baseline Profiling (DB-backed) + Drift Metrics + Figures (MERGED)

Tables (Excel):
- tables/RQ1_Table1.xlsx : Baseline descriptive statistics (numeric features)
- tables/RQ1_Table2.xlsx : Baseline vs New batch comparison (numeric features)
- tables/RQ1_Table3.xlsx : Drift metrics per feature (KS + PSI)

Figures (PDF):
- figures/RQ1_Fig1.pdf : Weekly_Sales distribution (baseline vs new batch)
- figures/RQ1_Fig2.pdf : Top drifted features by PSI (baseline vs new batch)
- figures/RQ1_Fig3.pdf : Mean PSI drift trend across batches (vs baseline)

Data source:
- PostgreSQL table: walmart_processed

Baseline:
- First N batches (default N=10)

New batch:
- Default: latest batch (MAX(Batch_ID))
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import text

# Make `src/` discoverable when running directly:
# python src/evaluation/rq1_baseline_profiling.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_ingestion.db_engine import get_engine  # noqa: E402


# -----------------------------
# Stats helpers
# -----------------------------
def ks_statistic(x_base: np.ndarray, x_new: np.ndarray) -> float:
    """Two-sample KS statistic (no SciPy required)."""
    x_base = np.sort(np.asarray(x_base, dtype=float))
    x_new = np.sort(np.asarray(x_new, dtype=float))

    if x_base.size == 0 or x_new.size == 0:
        return float("nan")

    data_all = np.sort(np.concatenate([x_base, x_new]))
    cdf_base = np.searchsorted(x_base, data_all, side="right") / x_base.size
    cdf_new = np.searchsorted(x_new, data_all, side="right") / x_new.size
    return float(np.max(np.abs(cdf_base - cdf_new)))


def psi(base: np.ndarray, new: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index using baseline quantile bins (robust)."""
    base = np.asarray(base, dtype=float)
    new = np.asarray(new, dtype=float)

    base = base[~np.isnan(base)]
    new = new[~np.isnan(new)]
    if base.size == 0 or new.size == 0:
        return float("nan")

    edges = np.quantile(base, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)

    # If edges collapse (near-constant baseline), return 0 drift
    if edges.size < 3:
        return 0.0

    base_counts, _ = np.histogram(base, bins=edges)
    new_counts, _ = np.histogram(new, bins=edges)

    base_pct = base_counts / max(base_counts.sum(), 1)
    new_pct = new_counts / max(new_counts.sum(), 1)

    eps = 1e-10
    base_pct = np.clip(base_pct, eps, None)
    new_pct = np.clip(new_pct, eps, None)

    return float(np.sum((new_pct - base_pct) * np.log(new_pct / base_pct)))


# -----------------------------
# RQ1 outputs
# -----------------------------
def build_rq1_tables(
    baseline_batches: int = 10,
    new_batch_id: int | None = None,
    output_dir: str = "tables",
) -> tuple[int, int]:
    """
    Generates RQ1_Table1/2/3 into output_dir.
    Returns: (baseline_batches, resolved_new_batch_id)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    engine = get_engine()
    df = pd.read_sql(text("SELECT * FROM walmart_processed"), con=engine)

    if "Batch_ID" not in df.columns:
        raise ValueError("Batch_ID not found in walmart_processed.")

    df["Batch_ID"] = pd.to_numeric(df["Batch_ID"], errors="coerce")
    df = df.dropna(subset=["Batch_ID"]).copy()
    df["Batch_ID"] = df["Batch_ID"].astype(int)

    if new_batch_id is None:
        new_batch_id = int(df["Batch_ID"].max())

    baseline_df = df[df["Batch_ID"] <= baseline_batches].copy()
    new_df = df[df["Batch_ID"] == new_batch_id].copy()

    if baseline_df.empty or new_df.empty:
        raise ValueError("Baseline or new batch is empty. Check Batch_ID logic.")

    numeric_cols = baseline_df.select_dtypes(include=["number"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ["Batch_ID"]]

    # -------------------------
    # Table1: baseline descriptive stats
    # -------------------------
    desc = baseline_df[numeric_cols].describe().T.reset_index().rename(columns={"index": "Feature"})
    table1_path = Path(output_dir) / "RQ1_Table1.xlsx"
    desc.to_excel(table1_path, index=False)

    # -------------------------
    # Table2: baseline vs new batch comparison
    # -------------------------
    rows = []
    for col in numeric_cols:
        b = pd.to_numeric(baseline_df[col], errors="coerce").dropna()
        n = pd.to_numeric(new_df[col], errors="coerce").dropna()
        if len(b) == 0 or len(n) == 0:
            continue

        rows.append(
            {
                "Feature": col,
                "Baseline_Mean": float(b.mean()),
                "NewBatch_Mean": float(n.mean()),
                "Baseline_Std": float(b.std()),
                "NewBatch_Std": float(n.std()),
                "Mean_Diff": float(n.mean() - b.mean()),
            }
        )
    comp_df = pd.DataFrame(rows)
    table2_path = Path(output_dir) / "RQ1_Table2.xlsx"
    comp_df.to_excel(table2_path, index=False)

    # -------------------------
    # Table3: drift metrics per feature (KS + PSI)
    # -------------------------
    drift_rows = []
    for col in numeric_cols:
        b = pd.to_numeric(baseline_df[col], errors="coerce").dropna().values
        n = pd.to_numeric(new_df[col], errors="coerce").dropna().values
        if len(b) == 0 or len(n) == 0:
            continue

        drift_rows.append(
            {
                "Feature": col,
                "KS_Statistic": ks_statistic(b, n),
                "PSI": psi(b, n, bins=10),
            }
        )

    drift_df = pd.DataFrame(drift_rows).sort_values("PSI", ascending=False)
    table3_path = Path(output_dir) / "RQ1_Table3.xlsx"
    drift_df.to_excel(table3_path, index=False)

    print("RQ1 tables generated successfully:")
    print(f"- {table1_path.as_posix()}")
    print(f"- {table2_path.as_posix()}")
    print(f"- {table3_path.as_posix()}")
    print(f"Baseline batches used: 1..{baseline_batches}")
    print(f"New batch compared: Batch_ID={new_batch_id}")

    return baseline_batches, int(new_batch_id)


def generate_rq1_figures(
    baseline_batches: int = 10,
    new_batch_id: int | None = None,
    output_dir: str = "figures",
) -> tuple[int, int]:
    """
    Generates RQ1_Fig1/2/3 into output_dir.
    Returns: (baseline_batches, resolved_new_batch_id)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    engine = get_engine()
    df = pd.read_sql(text("SELECT * FROM walmart_processed"), con=engine)

    if "Batch_ID" not in df.columns:
        raise ValueError("Batch_ID not found in walmart_processed.")

    df["Batch_ID"] = pd.to_numeric(df["Batch_ID"], errors="coerce")
    df = df.dropna(subset=["Batch_ID"]).copy()
    df["Batch_ID"] = df["Batch_ID"].astype(int)

    if new_batch_id is None:
        new_batch_id = int(df["Batch_ID"].max())

    baseline_df = df[df["Batch_ID"] <= baseline_batches].copy()
    new_df = df[df["Batch_ID"] == new_batch_id].copy()

    if baseline_df.empty or new_df.empty:
        raise ValueError("Baseline or new batch is empty. Check Batch_ID logic.")

    # -------------------------
    # Fig1: Weekly_Sales distribution shift
    # -------------------------
    fig1_path = Path(output_dir) / "RQ1_Fig1.pdf"
    plt.figure()
    plt.hist(baseline_df["Weekly_Sales"], bins=40, alpha=0.6, label=f"Baseline (Batches 1..{baseline_batches})")
    plt.hist(new_df["Weekly_Sales"], bins=40, alpha=0.6, label=f"New Batch ({new_batch_id})")
    plt.xlabel("Weekly_Sales")
    plt.ylabel("Frequency")
    plt.title("RQ1_Fig1: Weekly_Sales Distribution (Baseline vs New Batch)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig1_path)
    plt.close()

    # -------------------------
    # Fig2: Top features by PSI (baseline vs new)
    # -------------------------
    numeric_cols = baseline_df.select_dtypes(include=["number"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ["Batch_ID"]]

    psi_rows = []
    for col in numeric_cols:
        b = pd.to_numeric(baseline_df[col], errors="coerce").dropna().values
        n = pd.to_numeric(new_df[col], errors="coerce").dropna().values
        if len(b) == 0 or len(n) == 0:
            continue
        psi_rows.append({"Feature": col, "PSI": psi(b, n, bins=10)})

    psi_df = pd.DataFrame(psi_rows).sort_values("PSI", ascending=False)
    top = psi_df.head(10)

    fig2_path = Path(output_dir) / "RQ1_Fig2.pdf"
    plt.figure(figsize=(9, 4))
    plt.bar(top["Feature"], top["PSI"])
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Feature")
    plt.ylabel("PSI")
    plt.title("RQ1_Fig2: Top Drifted Features by PSI (Baseline vs New Batch)")
    plt.tight_layout()
    plt.savefig(fig2_path)
    plt.close()

    # -------------------------
    # Fig3: Drift trend over batches (mean PSI vs baseline)
    # -------------------------
    baseline_ref = baseline_df.copy()
    trend_rows = []

    for b_id, g in df.groupby("Batch_ID"):
        if int(b_id) <= baseline_batches:
            continue

        vals = []
        for col in numeric_cols:
            b = pd.to_numeric(baseline_ref[col], errors="coerce").dropna().values
            n = pd.to_numeric(g[col], errors="coerce").dropna().values
            if len(b) == 0 or len(n) == 0:
                continue
            vals.append(psi(b, n, bins=10))

        mean_psi = float(np.mean(vals)) if vals else np.nan
        trend_rows.append({"Batch_ID": int(b_id), "Mean_PSI": mean_psi})

    trend_df = pd.DataFrame(trend_rows).sort_values("Batch_ID")

    fig3_path = Path(output_dir) / "RQ1_Fig3.pdf"
    plt.figure()
    plt.plot(trend_df["Batch_ID"], trend_df["Mean_PSI"])
    plt.xlabel("Batch_ID (Weekly)")
    plt.ylabel("Mean PSI vs Baseline")
    plt.title("RQ1_Fig3: Drift Trend Across Batches (Mean PSI)")
    plt.tight_layout()
    plt.savefig(fig3_path)
    plt.close()

    print("RQ1 figures generated successfully:")
    print(f"- {fig1_path.as_posix()}")
    print(f"- {fig2_path.as_posix()}")
    print(f"- {fig3_path.as_posix()}")
    print(f"Baseline batches used: 1..{baseline_batches}")
    print(f"New batch compared: Batch_ID={new_batch_id}")

    return baseline_batches, int(new_batch_id)


def build_rq1_outputs(
    baseline_batches: int = 10,
    new_batch_id: int | None = None,
    table_dir: str = "tables",
    fig_dir: str = "figures",
) -> None:
    """
    One-shot RQ1 runner: generates ALL RQ1 tables + figures.
    """
    _, resolved_new = build_rq1_tables(baseline_batches=baseline_batches, new_batch_id=new_batch_id, output_dir=table_dir)
    generate_rq1_figures(baseline_batches=baseline_batches, new_batch_id=resolved_new, output_dir=fig_dir)


if __name__ == "__main__":
    build_rq1_outputs(baseline_batches=10, new_batch_id=None, table_dir="tables", fig_dir="figures")
