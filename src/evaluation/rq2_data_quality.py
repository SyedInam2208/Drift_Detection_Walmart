"""
RQ2 – Data Quality Monitoring (DB-backed) – Enhanced Table1 (Non-zero / Valuable)

Artifacts:
- tables/RQ2_Table1.xlsx : Feature Health & Drift Summary (curated, ranked)
- tables/RQ2_Table2.xlsx : Worst 10 batches + global summary stats
- figures/RQ2_Fig1.pdf   : Quality Heatmap for top columns (baseline vs new)
- figures/RQ2_Fig2.pdf   : Variance drift over time (top 3 numeric features)
- figures/RQ2_Fig3.pdf   : Outlier frequency trend (batch-level)

Data source:
- PostgreSQL table: walmart_processed
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import text

# Allow direct execution: python -m src.evaluation.rq2_data_quality
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_ingestion.db_engine import get_engine  # noqa: E402


# ---------------------------
# Helpers
# ---------------------------

def numeric_cols(df: pd.DataFrame) -> list[str]:
    cols = df.select_dtypes(include=["number"]).columns.tolist()
    return [c for c in cols if c != "Batch_ID"]


def outlier_rate_z(series: pd.Series, z_thresh: float = 3.0) -> float:
    """Percent of values with |z| > z_thresh (numeric)."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return float("nan")
    mu = s.mean()
    sd = s.std() + 1e-9
    z = (s - mu) / sd
    return float((np.abs(z) > z_thresh).mean() * 100)


def duplicate_row_rate(df: pd.DataFrame) -> float:
    if df.empty:
        return float("nan")
    return float(df.duplicated().mean() * 100)


def psi(base: np.ndarray, new: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index using baseline quantile bins (robust, no SciPy).
    """
    base = np.asarray(base, dtype=float)
    new = np.asarray(new, dtype=float)

    base = base[~np.isnan(base)]
    new = new[~np.isnan(new)]
    if base.size == 0 or new.size == 0:
        return float("nan")

    edges = np.quantile(base, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)

    # If baseline is almost constant, bins collapse; treat as no drift
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


def column_health_and_drift_metrics(
    base: pd.DataFrame,
    new: pd.DataFrame,
    col: str,
    z_thresh: float = 3.0,
) -> dict:
    """
    Computes a richer set of column-level metrics so Table1 stays meaningful even
    when missing/outliers are low.
    """

    # Missing (works for all types)
    b_miss = float(base[col].isna().mean() * 100) if len(base) else np.nan
    n_miss = float(new[col].isna().mean() * 100) if len(new) else np.nan
    miss_diff_pp = (n_miss - b_miss) if (not np.isnan(b_miss) and not np.isnan(n_miss)) else np.nan

    is_num = pd.api.types.is_numeric_dtype(base[col])
    col_type = "numeric" if is_num else "categorical"

    # Defaults
    b_out = n_out = np.nan
    b_zero = n_zero = np.nan
    b_mean = n_mean = np.nan
    b_std = n_std = np.nan
    mean_diff_pct = np.nan
    std_diff_pct = np.nan
    psi_val = np.nan

    b_unique = n_unique = np.nan
    unique_diff = np.nan

    if is_num:
        b_series = pd.to_numeric(base[col], errors="coerce")
        n_series = pd.to_numeric(new[col], errors="coerce")

        b_out = outlier_rate_z(b_series, z_thresh=z_thresh)
        n_out = outlier_rate_z(n_series, z_thresh=z_thresh)

        b_zero = float((b_series == 0).mean() * 100) if len(base) else np.nan
        n_zero = float((n_series == 0).mean() * 100) if len(new) else np.nan

        b_mean = float(b_series.dropna().mean()) if not b_series.dropna().empty else np.nan
        n_mean = float(n_series.dropna().mean()) if not n_series.dropna().empty else np.nan

        b_std = float(b_series.dropna().std()) if not b_series.dropna().empty else np.nan
        n_std = float(n_series.dropna().std()) if not n_series.dropna().empty else np.nan

        # % shifts (robust denom)
        denom_mean = abs(b_mean) if (not np.isnan(b_mean) and abs(b_mean) > 1e-9) else np.nan
        mean_diff_pct = float((n_mean - b_mean) / denom_mean * 100) if (not np.isnan(denom_mean) and not np.isnan(n_mean)) else np.nan

        denom_std = abs(b_std) if (not np.isnan(b_std) and abs(b_std) > 1e-9) else np.nan
        std_diff_pct = float((n_std - b_std) / denom_std * 100) if (not np.isnan(denom_std) and not np.isnan(n_std)) else np.nan

        psi_val = psi(b_series.values, n_series.values, bins=10)

        out_diff_pp = (n_out - b_out) if (not np.isnan(b_out) and not np.isnan(n_out)) else np.nan
        zero_diff_pp = (n_zero - b_zero) if (not np.isnan(b_zero) and not np.isnan(n_zero)) else np.nan

    else:
        # categorical stability signal
        b_unique = int(base[col].nunique(dropna=True)) if len(base) else np.nan
        n_unique = int(new[col].nunique(dropna=True)) if len(new) else np.nan
        unique_diff = float(n_unique - b_unique) if (not np.isnan(b_unique) and not np.isnan(n_unique)) else np.nan

        out_diff_pp = np.nan
        zero_diff_pp = np.nan

    # SignalScore ranks columns by "meaningful change"
    # PSI scaled *100; mean/std already in %, missing/outlier/zero are pp differences.
    parts = [
        abs(miss_diff_pp) if not np.isnan(miss_diff_pp) else 0.0,
        abs(out_diff_pp) if not np.isnan(out_diff_pp) else 0.0,
        abs(zero_diff_pp) if not np.isnan(zero_diff_pp) else 0.0,
        abs(mean_diff_pct) if not np.isnan(mean_diff_pct) else 0.0,
        abs(std_diff_pct) if not np.isnan(std_diff_pct) else 0.0,
        (psi_val * 100) if not np.isnan(psi_val) else 0.0,
        abs(unique_diff) if not np.isnan(unique_diff) else 0.0,
    ]
    signal_score = float(np.sum(parts))

    return {
        "Column": col,
        "Type": col_type,

        "Baseline_Missing_%": b_miss,
        "New_Missing_%": n_miss,
        "Missing_Diff_pp": miss_diff_pp,

        "Baseline_Outlier_%": b_out,
        "New_Outlier_%": n_out,
        "Outlier_Diff_pp": out_diff_pp,

        "Baseline_Zero_%": b_zero,
        "New_Zero_%": n_zero,
        "Zero_Diff_pp": zero_diff_pp,

        "Baseline_Mean": b_mean,
        "New_Mean": n_mean,
        "Mean_Diff_%": mean_diff_pct,

        "Baseline_Std": b_std,
        "New_Std": n_std,
        "Std_Diff_%": std_diff_pct,

        "Baseline_Unique": b_unique,
        "New_Unique": n_unique,
        "Unique_Diff": unique_diff,

        "PSI": psi_val,
        "SignalScore": signal_score,
    }


def build_rq2_table1(
    df: pd.DataFrame,
    baseline_batches: int,
    new_batch_id: int,
    z_thresh: float = 3.0,
    min_signal: float = 2.0,
    top_k: int = 15,
) -> pd.DataFrame:
    """
    Curated, high-value Table1:
    - Computes rich column metrics vs baseline
    - Filters out "boring" columns (SignalScore < min_signal)
    - Always returns something (fallback: top_k)
    """
    base = df[df["Batch_ID"] <= baseline_batches].copy()
    new = df[df["Batch_ID"] == new_batch_id].copy()

    cols = [c for c in df.columns if c != "Batch_ID"]

    rows: list[dict] = []
    for col in cols:
        try:
            rows.append(column_health_and_drift_metrics(base, new, col, z_thresh=z_thresh))
        except Exception:
            continue

    t1 = pd.DataFrame(rows).sort_values("SignalScore", ascending=False)

    filtered = t1[t1["SignalScore"] >= float(min_signal)].copy()
    if filtered.empty:
        filtered = t1.head(top_k).copy()
    else:
        filtered = filtered.head(top_k).copy()

    return filtered.reset_index(drop=True)


def batch_scoreboard(df: pd.DataFrame, z_thresh: float = 3.0) -> pd.DataFrame:
    """
    Batch-level quality scoreboard:
    - Missing cells (%)
    - Avg outlier rate across numeric columns (%)
    - Duplicate row rate (%)
    - Composite risk score
    """
    num = numeric_cols(df)

    rows = []
    for b, g in df.groupby("Batch_ID"):
        b = int(b)
        g = g.copy()

        total_cells = g.shape[0] * g.shape[1]
        missing_cells = int(g.isna().sum().sum())
        miss_pct = (missing_cells / total_cells * 100) if total_cells > 0 else 0.0

        out_rates = [outlier_rate_z(g[c], z_thresh=z_thresh) for c in num]
        out_pct = float(np.nanmean(out_rates)) if out_rates else np.nan

        dup_pct = duplicate_row_rate(g)

        risk = miss_pct + (out_pct if not np.isnan(out_pct) else 0.0) + (dup_pct if not np.isnan(dup_pct) else 0.0)

        drivers = {
            "Missing": miss_pct,
            "Outliers": out_pct if not np.isnan(out_pct) else 0.0,
            "Duplicates": dup_pct if not np.isnan(dup_pct) else 0.0,
        }
        top_driver = max(drivers, key=drivers.get)

        rows.append(
            {
                "Batch_ID": b,
                "Rows": int(g.shape[0]),
                "MissingCells_%": float(miss_pct),
                "AvgOutlier_%": float(out_pct) if not np.isnan(out_pct) else np.nan,
                "DuplicateRow_%": float(dup_pct) if not np.isnan(dup_pct) else np.nan,
                "Batch_RiskScore": float(risk),
                "Primary_Issue": top_driver,
            }
        )

    return pd.DataFrame(rows).sort_values("Batch_ID")


def variance_drift(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    cols = numeric_cols(df)
    rows = []
    for b, g in df.groupby("Batch_ID"):
        row = {"Batch_ID": int(b)}
        for c in cols:
            row[c] = float(pd.to_numeric(g[c], errors="coerce").var())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Batch_ID"), cols


# ---------------------------
# Main generator
# ---------------------------

def generate_rq2_outputs(
    baseline_batches: int = 10,
    new_batch_id: int | None = None,
    table_dir: str = "tables",
    fig_dir: str = "figures",
    z_thresh: float = 3.0,
) -> None:
    Path(table_dir).mkdir(parents=True, exist_ok=True)
    Path(fig_dir).mkdir(parents=True, exist_ok=True)

    engine = get_engine()
    df = pd.read_sql(text("SELECT * FROM walmart_processed"), con=engine)

    if "Batch_ID" not in df.columns:
        raise ValueError("Batch_ID not found in walmart_processed table.")

    df["Batch_ID"] = pd.to_numeric(df["Batch_ID"], errors="coerce")
    df = df.dropna(subset=["Batch_ID"]).copy()
    df["Batch_ID"] = df["Batch_ID"].astype(int)

    if new_batch_id is None:
        new_batch_id = int(df["Batch_ID"].max())

    # ---------------------------
    # UPDATED RQ2_Table1
    # ---------------------------
    t1 = build_rq2_table1(
        df,
        baseline_batches=baseline_batches,
        new_batch_id=new_batch_id,
        z_thresh=z_thresh,
        min_signal=2.0,
        top_k=15,
    )
    t1_path = Path(table_dir) / "RQ2_Table1.xlsx"
    t1.to_excel(t1_path, index=False)

    # ---------------------------
    # RQ2_Table2: Worst 10 batches + summary
    # ---------------------------
    batch = batch_scoreboard(df, z_thresh=z_thresh)
    worst10 = batch.sort_values("Batch_RiskScore", ascending=False).head(10).copy()

    summary = pd.DataFrame(
        [
            {
                "Metric": "Batch_RiskScore",
                "Mean": float(batch["Batch_RiskScore"].mean()),
                "Std": float(batch["Batch_RiskScore"].std()),
                "P50": float(batch["Batch_RiskScore"].quantile(0.50)),
                "P95": float(batch["Batch_RiskScore"].quantile(0.95)),
                "Max": float(batch["Batch_RiskScore"].max()),
            }
        ]
    )

    t2_path = Path(table_dir) / "RQ2_Table2.xlsx"
    with pd.ExcelWriter(t2_path, engine="openpyxl") as writer:
        worst10.to_excel(writer, sheet_name="Worst_10_Batches", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)

    # ---------------------------
    # RQ2_Fig1: Heatmap for top columns in Table1 (numeric only)
    # ---------------------------
    base = df[df["Batch_ID"] <= baseline_batches]
    new = df[df["Batch_ID"] == new_batch_id]

    heat_cols = [c for c in t1["Column"].tolist() if c in numeric_cols(df)]
    heat_cols = heat_cols[:10]  # keep readable

    if not heat_cols:
        heat_cols = numeric_cols(df)[:10]

    row_names = [
        "Base Missing%",
        "New Missing%",
        "Base Outlier%",
        "New Outlier%",
        "Mean Diff %",
        "Std Diff %",
        "PSI",
    ]

    mat = []
    for c in heat_cols:
        # reuse the same metric builder for consistency
        m = column_health_and_drift_metrics(base, new, c, z_thresh=z_thresh)
        mat.append(
            [
                m["Baseline_Missing_%"],
                m["New_Missing_%"],
                m["Baseline_Outlier_%"],
                m["New_Outlier_%"],
                m["Mean_Diff_%"],
                m["Std_Diff_%"],
                m["PSI"],
            ]
        )

    mat = np.array(mat, dtype=float).T  # metrics x columns

    fig1_path = Path(fig_dir) / "RQ2_Fig1.pdf"
    plt.figure(figsize=(12, 4))
    plt.imshow(mat, aspect="auto")
    plt.yticks(np.arange(len(row_names)), row_names)
    plt.xticks(np.arange(len(heat_cols)), heat_cols, rotation=45, ha="right")
    plt.title("RQ2_Fig1: Quality + Drift Heatmap (Top Columns, Baseline vs New Batch)")
    plt.colorbar(label="Value")
    plt.tight_layout()
    plt.savefig(fig1_path)
    plt.close()

    # ---------------------------
    # RQ2_Fig2: Variance drift (top 3 numeric features)
    # ---------------------------
    var_df, cols = variance_drift(df)

    fig2_path = Path(fig_dir) / "RQ2_Fig2.pdf"
    plt.figure(figsize=(9, 4))
    for c in cols[:3]:
        plt.plot(var_df["Batch_ID"], var_df[c], label=c)
    plt.xlabel("Batch_ID (Weekly)")
    plt.ylabel("Variance")
    plt.title("RQ2_Fig2: Variance Drift Over Time (Top 3 Numeric Features)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig2_path)
    plt.close()

    # ---------------------------
    # RQ2_Fig3: Outlier trend (batch-level)
    # ---------------------------
    fig3_path = Path(fig_dir) / "RQ2_Fig3.pdf"
    plt.figure()
    plt.plot(batch["Batch_ID"], batch["AvgOutlier_%"])
    plt.xlabel("Batch_ID (Weekly)")
    plt.ylabel("Avg Outliers (%)")
    plt.title(f"RQ2_Fig3: Outlier Frequency Trend (z > {z_thresh})")
    plt.tight_layout()
    plt.savefig(fig3_path)
    plt.close()

    print("RQ2 outputs generated successfully:")
    print(f"- {t1_path.as_posix()}")
    print(f"- {t2_path.as_posix()}")
    print(f"- {fig1_path.as_posix()}")
    print(f"- {fig2_path.as_posix()}")
    print(f"- {fig3_path.as_posix()}")
    print(f"Baseline batches used: 1..{baseline_batches}")
    print(f"New batch compared: Batch_ID={new_batch_id}")


if __name__ == "__main__":
    generate_rq2_outputs(
        baseline_batches=10,
        new_batch_id=None,
        table_dir="tables",
        fig_dir="figures",
        z_thresh=3.0,
    )
