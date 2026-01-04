"""
RQ3 – Impact of Data Drift on Analytical Outputs (DB-backed)

Goal:
Show how drift/stability in processed data impacts downstream analytical performance.

Approach:
- Use a simple and defensible baseline predictor: rolling mean forecast on Weekly_Sales.
- Compute error metrics (MAE/RMSE) per batch and compare baseline window vs latest batch.
- Compute drift proxy vs baseline using Mean PSI across numeric features (no SciPy needed).
- Analyze relationship between drift magnitude and error metrics.

Artifacts generated (5 total):
Tables:
- tables/RQ3_Table1.xlsx : Baseline vs New batch error summary (MAE, RMSE, N)
- tables/RQ3_Table2.xlsx : Batch-wise metrics (MAE, RMSE, Mean_PSI_Drift)

Figures:
- figures/RQ3_Fig1.pdf   : Batch-wise MAE and RMSE over time
- figures/RQ3_Fig2.pdf   : Scatter: Mean PSI drift vs MAE (with correlation shown in title)
- figures/RQ3_Fig3.pdf   : Error distribution comparison (Baseline vs New batch)

Data source:
- PostgreSQL table: walmart_processed
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import text

# Make src visible when running as a script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_ingestion.db_engine import get_engine  # noqa: E402


# -----------------------------
# Drift metric: PSI (no SciPy)
# -----------------------------
def psi(base: np.ndarray, new: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index using baseline quantile bins.
    Returns 0 for constant features (no meaningful binning).
    """
    base = np.asarray(base)
    new = np.asarray(new)

    if base.size == 0 or new.size == 0:
        return float("nan")

    edges = np.quantile(base, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
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


def mean_psi_drift_vs_baseline(
    baseline_df: pd.DataFrame,
    batch_df: pd.DataFrame,
    numeric_cols: list[str],
    bins: int = 10
) -> float:
    vals = []
    for col in numeric_cols:
        b = pd.to_numeric(baseline_df[col], errors="coerce").dropna().values
        n = pd.to_numeric(batch_df[col], errors="coerce").dropna().values
        if len(b) == 0 or len(n) == 0:
            continue
        vals.append(psi(b, n, bins=bins))
    return float(np.mean(vals)) if vals else np.nan


# -----------------------------
# Forecast proxy: rolling mean
# -----------------------------
def build_batch_forecast_errors(
    df: pd.DataFrame,
    target_col: str = "Weekly_Sales",
    rolling_window: int = 4
) -> pd.DataFrame:
    """
    Compute per-row forecast and error using a rolling mean.
    Forecast is computed across the whole dataset in batch order.

    Output contains:
    - Forecast
    - Error
    - AbsError
    - SqError
    """
    df = df.sort_values(["Batch_ID"]).copy()
    y = pd.to_numeric(df[target_col], errors="coerce")

    df["Forecast"] = y.rolling(window=rolling_window, min_periods=rolling_window).mean()
    df["Error"] = y - df["Forecast"]
    df["AbsError"] = df["Error"].abs()
    df["SqError"] = df["Error"] ** 2

    # drop rows where forecast isn't available yet
    df = df.dropna(subset=["Forecast", "Error"])
    return df


def summarize_errors_by_batch(err_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for b, g in err_df.groupby("Batch_ID"):
        rows.append({
            "Batch_ID": int(b),
            "N": int(len(g)),
            "MAE": float(g["AbsError"].mean()),
            "RMSE": float(np.sqrt(g["SqError"].mean()))
        })
    return pd.DataFrame(rows).sort_values("Batch_ID")


def pearson_corr(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    m = x.notna() & y.notna()
    if m.sum() < 3:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])


# -----------------------------
# Main generator
# -----------------------------
def generate_rq3_outputs(
    baseline_batches: int = 10,
    new_batch_id: int | None = None,
    rolling_window: int = 4,
    psi_bins: int = 10,
    table_dir: str = "tables",
    fig_dir: str = "figures",
) -> None:
    Path(table_dir).mkdir(parents=True, exist_ok=True)
    Path(fig_dir).mkdir(parents=True, exist_ok=True)

    # Load processed data from PostgreSQL
    engine = get_engine()
    df = pd.read_sql(text("SELECT * FROM walmart_processed"), con=engine)

    if "Batch_ID" not in df.columns:
        raise ValueError("Batch_ID not found in walmart_processed.")

    df["Batch_ID"] = pd.to_numeric(df["Batch_ID"], errors="coerce")

    if new_batch_id is None:
        new_batch_id = int(df["Batch_ID"].max())

    if "Weekly_Sales" not in df.columns:
        raise ValueError("Weekly_Sales not found in walmart_processed. Check ingestion script.")

    # Build baseline/new subsets for drift calculations
    baseline_ref = df[df["Batch_ID"] <= baseline_batches].copy()
    new_batch_df = df[df["Batch_ID"] == new_batch_id].copy()

    if baseline_ref.empty:
        raise ValueError("Baseline reference is empty. Increase baseline_batches or verify Batch_ID.")
    if new_batch_df.empty:
        raise ValueError(f"New batch is empty for Batch_ID={new_batch_id}.")

    # Select numeric columns for drift computation (exclude Batch_ID)
    numeric_cols = baseline_ref.select_dtypes(include=["number"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "Batch_ID"]

    # -----------------------------
    # Compute proxy forecast errors
    # -----------------------------
    err_df = build_batch_forecast_errors(df, target_col="Weekly_Sales", rolling_window=rolling_window)
    batch_err = summarize_errors_by_batch(err_df)

    # -----------------------------
    # Compute drift (Mean PSI) per batch vs baseline
    # -----------------------------
    drift_rows = []
    for b, g in df.groupby("Batch_ID"):
        b = int(b)
        if b <= baseline_batches:
            continue
        mean_psi_val = mean_psi_drift_vs_baseline(
            baseline_df=baseline_ref,
            batch_df=g,
            numeric_cols=numeric_cols,
            bins=psi_bins
        )
        drift_rows.append({"Batch_ID": b, "Mean_PSI_Drift": mean_psi_val})

    drift_df = pd.DataFrame(drift_rows).sort_values("Batch_ID")

    # Merge batch-wise error metrics with drift metrics
    merged = batch_err.merge(drift_df, on="Batch_ID", how="left")

    # -----------------------------
    # RQ3_Table1: Baseline vs New batch summary
    # -----------------------------
    base_err = err_df[err_df["Batch_ID"] <= baseline_batches]
    new_err = err_df[err_df["Batch_ID"] == new_batch_id]

    t1 = pd.DataFrame([
        {
            "Window": f"Baseline (Batches 1..{baseline_batches})",
            "N": int(len(base_err)),
            "MAE": float(base_err["AbsError"].mean()) if len(base_err) else np.nan,
            "RMSE": float(np.sqrt(base_err["SqError"].mean())) if len(base_err) else np.nan,
        },
        {
            "Window": f"New Batch (Batch_ID={new_batch_id})",
            "N": int(len(new_err)),
            "MAE": float(new_err["AbsError"].mean()) if len(new_err) else np.nan,
            "RMSE": float(np.sqrt(new_err["SqError"].mean())) if len(new_err) else np.nan,
        }
    ])

    t1_path = Path(table_dir) / "RQ3_Table1.xlsx"
    t1.to_excel(t1_path, index=False)

    # -----------------------------
    # RQ3_Table2: Batch-wise summary (errors + drift)
    # -----------------------------
    t2_path = Path(table_dir) / "RQ3_Table2.xlsx"
    merged.to_excel(t2_path, index=False)

    # -----------------------------
    # RQ3_Fig1: Error metrics over time
    # -----------------------------
    fig1_path = Path(fig_dir) / "RQ3_Fig1.pdf"
    plt.figure(figsize=(9, 4))
    plt.plot(merged["Batch_ID"], merged["MAE"], label="MAE")
    plt.plot(merged["Batch_ID"], merged["RMSE"], label="RMSE")
    plt.xlabel("Batch_ID (Weekly)")
    plt.ylabel("Error")
    plt.title("RQ3_Fig1: Forecast Error Metrics Over Time (Rolling Mean Proxy)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig1_path)
    plt.close()

    # -----------------------------
    # RQ3_Fig2: Drift vs Error correlation
    # -----------------------------
    fig2_path = Path(fig_dir) / "RQ3_Fig2.pdf"
    corr = pearson_corr(merged["Mean_PSI_Drift"], merged["MAE"])

    plt.figure()
    plt.scatter(merged["Mean_PSI_Drift"], merged["MAE"])
    plt.xlabel("Mean PSI Drift vs Baseline")
    plt.ylabel("MAE")
    title = "RQ3_Fig2: Drift Magnitude vs MAE"
    if not np.isnan(corr):
        title += f" (Pearson r={corr:.3f})"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fig2_path)
    plt.close()

    # -----------------------------
    # RQ3_Fig3: Error distribution comparison (baseline vs new)
    # -----------------------------
    fig3_path = Path(fig_dir) / "RQ3_Fig3.pdf"
    plt.figure()
    plt.hist(base_err["Error"], bins=40, alpha=0.6, label=f"Baseline 1..{baseline_batches}")
    plt.hist(new_err["Error"], bins=40, alpha=0.6, label=f"New Batch {new_batch_id}")
    plt.xlabel("Forecast Error (Actual - Forecast)")
    plt.ylabel("Frequency")
    plt.title("RQ3_Fig3: Error Distribution (Baseline vs New Batch)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig3_path)
    plt.close()

    # -----------------------------
    # Console output
    # -----------------------------
    print("RQ3 outputs generated successfully:")
    print(f"- {t1_path}")
    print(f"- {t2_path}")
    print(f"- {fig1_path}")
    print(f"- {fig2_path}")
    print(f"- {fig3_path}")
    print(f"Baseline batches used: 1..{baseline_batches}")
    print(f"New batch compared: Batch_ID={new_batch_id}")
    print(f"Rolling window used (weeks): {rolling_window}")
    print(f"PSI bins: {psi_bins}")


if __name__ == "__main__":
    generate_rq3_outputs(
        baseline_batches=10,
        new_batch_id=None,
        rolling_window=4,
        psi_bins=10,
        table_dir="tables",
        fig_dir="figures",
    )

