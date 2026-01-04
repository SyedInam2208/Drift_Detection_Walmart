"""
RQ4 – Measurable impact of implementing drift detection on reliability/stability
(DB-backed, uses walmart_processed)

We quantify impact by comparing two scenarios:
A) No drift detection: downstream uses every batch output.
B) With drift detection: batches flagged as drifted trigger "fallback" behavior:
   - Dashboards/analytics: replace KPI values with last stable KPI (hold-last-good).

Improvements vs previous version:
- Drift computed on INPUT features only (excludes Weekly_Sales and identifiers).
- Drift score per batch = median PSI across selected features (robust).
- Threshold selectable: WarmupQuantile(target=10%) (default) or Median+k*MAD (robust).
- Table2 written as two sheets: Alert_Only + Full_Log (report-friendly).

Artifacts generated (5):
- tables/RQ4_Table1.xlsx  (stability metrics: Raw vs Stabilized)
- tables/RQ4_Table2.xlsx  (alert log: Alert_Only + Full_Log sheets)
- figures/RQ4_Fig1.pdf    (KPI stability line chart: raw vs stabilized)
- figures/RQ4_Fig2.pdf    (drift score + threshold + alert markers)
- figures/RQ4_Fig3.pdf    (bar chart: stability improvements)

Author: (your project)
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from sqlalchemy import text

# Make src visible when running as a script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Local DB engine helper (provided in your repo)
from data_ingestion.db_engine import get_engine


# -----------------------------
# PSI + robust helpers
# -----------------------------
def _to_bins(values: np.ndarray, bins: int = 10) -> np.ndarray:
    """Create bin edges using quantiles (robust) with fallback to linspace."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.array([0.0, 1.0])

    # quantile edges
    qs = np.linspace(0, 1, bins + 1)
    edges = np.quantile(values, qs)

    # if edges collapse (constant-ish), fallback
    if np.unique(edges).size < 3:
        mn, mx = float(np.min(values)), float(np.max(values))
        if mn == mx:
            return np.array([mn - 0.5, mx + 0.5])
        edges = np.linspace(mn, mx, bins + 1)

    # ensure strictly increasing edges
    edges = np.unique(edges)
    if edges.size < 2:
        edges = np.array([float(values.min()) - 0.5, float(values.max()) + 0.5])
    return edges


def psi(base: np.ndarray, new: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between baseline and new distribution."""
    base = np.asarray(base, dtype=float)
    new = np.asarray(new, dtype=float)

    base = base[~np.isnan(base)]
    new = new[~np.isnan(new)]

    if len(base) == 0 or len(new) == 0:
        return np.nan

    edges = _to_bins(base, bins=bins)

    base_counts, _ = np.histogram(base, bins=edges)
    new_counts, _ = np.histogram(new, bins=edges)

    base_pct = base_counts / max(base_counts.sum(), 1)
    new_pct = new_counts / max(new_counts.sum(), 1)

    # avoid log(0)
    eps = 1e-10
    base_pct = np.clip(base_pct, eps, None)
    new_pct = np.clip(new_pct, eps, None)

    return float(np.sum((new_pct - base_pct) * np.log(new_pct / base_pct)))


def mad(x: pd.Series | np.ndarray) -> float:
    """Median Absolute Deviation (MAD) with nan-safety."""
    x = np.asarray(pd.Series(x).dropna().values, dtype=float)
    if len(x) == 0:
        return 0.0
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def robust_drift_score_vs_baseline(
    baseline_df: pd.DataFrame,
    batch_df: pd.DataFrame,
    feature_cols: list[str],
    bins: int = 10
) -> float:
    """
    Compute drift score per batch = median PSI across selected features (robust).
    """
    vals = []
    for col in feature_cols:
        b = pd.to_numeric(baseline_df[col], errors="coerce").dropna().values
        n = pd.to_numeric(batch_df[col], errors="coerce").dropna().values
        if len(b) == 0 or len(n) == 0:
            continue
        vals.append(psi(b, n, bins=bins))
    return float(np.nanmedian(vals)) if len(vals) else np.nan


def select_input_features(df: pd.DataFrame) -> list[str]:
    """
    Select numeric 'input' features for drift detection, excluding target/KPI and identifiers.
    We intentionally exclude Weekly_Sales because it is:
      - the target / KPI used in downstream metrics
      - impacted by drift but not an "input feature" drift signal in this project design

    Exclusions include identifiers like Store/Dept/Date/Batch_ID if present.
    """
    exclude = {
        "Weekly_Sales",
        "Date",
        "Year",
        "Month",
        "Week",
        "Year_Week",
        "Batch_ID",
        "Store",
        "Dept",
        "Type",  # could be categorical
        "IsHoliday"  # could be boolean
    }
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude]

    # Additional guard: drop obvious ID-ish columns
    for c in list(feature_cols):
        if c.lower().endswith("_id") or c.lower() in {"id"}:
            feature_cols.remove(c)

    return feature_cols


# -----------------------------
# KPI impact logic
# -----------------------------
def build_kpi_by_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate simple KPIs per batch (for stability analysis)."""
    out = (
        df.groupby("Batch_ID", as_index=False)
        .agg(
            AvgWeeklySales=("Weekly_Sales", "mean"),
            TotalWeeklySales=("Weekly_Sales", "sum"),
            Count=("Weekly_Sales", "size"),
        )
        .sort_values("Batch_ID")
    )
    return out


def hold_last_stable(series: pd.Series, alerts: pd.Series) -> pd.Series:
    """
    If alert=True at batch t, replace KPI with last stable value (hold-last-good).
    """
    stabilized = []
    last_good = None
    for v, a in zip(series.values, alerts.values):
        if not a:
            last_good = v
            stabilized.append(v)
        else:
            stabilized.append(last_good if last_good is not None else v)
    return pd.Series(stabilized, index=series.index)


def stability_metrics_from_series(
    batch_ids: pd.Series,
    values: pd.Series,
    baseline_end: int,
    tolerance_pct: float = 5.0
) -> dict:
    """
    Compute reliability/stability metrics for KPI:
      - Baseline mean (batches 1..baseline_end)
      - % batches within tolerance band around baseline mean (post-baseline)
      - Mean absolute deviation from baseline mean (post-baseline)
    """
    df_tmp = pd.DataFrame({"Batch_ID": batch_ids, "Value": values}).dropna()
    baseline = df_tmp[df_tmp["Batch_ID"] <= baseline_end]
    post = df_tmp[df_tmp["Batch_ID"] > baseline_end]

    baseline_mean = float(baseline["Value"].mean()) if len(baseline) else float(df_tmp["Value"].mean())

    tol = abs(baseline_mean) * (tolerance_pct / 100.0)
    lower, upper = baseline_mean - tol, baseline_mean + tol

    within = post["Value"].between(lower, upper).mean() * 100.0 if len(post) else np.nan
    mad_from_base = float(np.mean(np.abs(post["Value"] - baseline_mean))) if len(post) else np.nan

    return {
        "BaselineMean": baseline_mean,
        "ToleranceBand": f"±{tolerance_pct:.1f}%",
        "WithinTolerance_%": float(within) if within == within else np.nan,
        "MeanAbsDeviationFromBaseline": mad_from_base
    }


# -----------------------------
# Main RQ4 runner
# -----------------------------
def generate_rq4_outputs(
    baseline_batches: int = 10,
    warmup_batches: int = 20,
    psi_bins: int = 10,
    tolerance_pct: float = 5.0,
    table_dir: str = "tables",
    fig_dir: str = "figures",
    threshold_method: str = "WarmupQuantile",
    target_alert_rate: float = 0.10,
    mad_k: float = 3.0,
) -> None:
    Path(table_dir).mkdir(parents=True, exist_ok=True)
    Path(fig_dir).mkdir(parents=True, exist_ok=True)

    engine = get_engine()
    df = pd.read_sql(text("SELECT * FROM walmart_processed"), con=engine)

    if "Batch_ID" not in df.columns or "Weekly_Sales" not in df.columns:
        raise ValueError("Required columns missing: Batch_ID and Weekly_Sales must exist in walmart_processed.")

    df["Batch_ID"] = df["Batch_ID"].astype(int)

    # Baseline reference for drift
    baseline_ref = df[df["Batch_ID"] <= baseline_batches].copy()
    if baseline_ref.empty:
        raise ValueError("Baseline reference is empty. Increase baseline_batches or verify Batch_ID values.")

    # Input features (robust + interpretable)
    feature_cols = select_input_features(df)
    if len(feature_cols) == 0:
        raise ValueError("No numeric input features left after exclusions. Review select_input_features().")

    # Drift score per batch (median PSI across input features vs baseline)
    drift_rows = []
    for b, g in df.groupby("Batch_ID"):
        score = robust_drift_score_vs_baseline(
            baseline_df=baseline_ref,
            batch_df=g,
            feature_cols=feature_cols,
            bins=psi_bins
        )
        drift_rows.append({"Batch_ID": int(b), "DriftScore_MedianPSI": score})

    drift_df = pd.DataFrame(drift_rows).sort_values("Batch_ID").reset_index(drop=True)

    # -----------------------------
    # Thresholding (selectable)
    # -----------------------------
    warmup_start = baseline_batches + 1
    warmup_end = baseline_batches + warmup_batches

    warmup_scores = drift_df.loc[
        (drift_df["Batch_ID"] >= warmup_start) & (drift_df["Batch_ID"] <= warmup_end),
        "DriftScore_MedianPSI"
    ].dropna()

    # Fallback if warmup window is too small
    if len(warmup_scores) < 5:
        warmup_scores = drift_df["DriftScore_MedianPSI"].dropna()

    if len(warmup_scores) == 0:
        raise ValueError("No warmup drift scores available to calibrate threshold.")

    method = str(threshold_method).strip().lower()

    if method == "warmupquantile":
        # Calibrate to a target alert rate on warmup scores.
        # target_alert_rate=0.10 => threshold = 90th percentile (1 - 0.10).
        q = 1.0 - float(target_alert_rate)
        q = min(max(q, 0.50), 0.99)  # safety clamp
        threshold = float(np.quantile(warmup_scores, q))
        threshold_label = f"WarmupQuantile(target={target_alert_rate:.0%})"

    elif method in {"mad", "median+mad", "median_mad"}:
        med = float(np.median(warmup_scores))
        dispersion = mad(warmup_scores)
        if np.isnan(dispersion) or dispersion == 0.0:
            # fallback if MAD collapses (very stable scores)
            dispersion = float(warmup_scores.std())
        threshold = med + float(mad_k) * float(dispersion)
        threshold_label = f"Median+{mad_k:g}*MAD"

    else:
        raise ValueError(
            f"Unknown threshold_method: {threshold_method}. Use 'WarmupQuantile' or 'MAD'."
        )

    drift_df["IsAlert"] = drift_df["DriftScore_MedianPSI"] > threshold

    # Downstream KPI impact (dashboard/analytics)
    kpi_df = build_kpi_by_batch(df)

    alert_map = drift_df.set_index("Batch_ID")["IsAlert"].to_dict()
    kpi_df["IsAlert"] = kpi_df["Batch_ID"].map(lambda b: bool(alert_map.get(int(b), False)))

    kpi_df["AvgWeeklySales_Stabilized"] = hold_last_stable(kpi_df["AvgWeeklySales"], kpi_df["IsAlert"])
    kpi_df["TotalWeeklySales_Stabilized"] = hold_last_stable(kpi_df["TotalWeeklySales"], kpi_df["IsAlert"])

    # Stability metrics (Raw vs Stabilized)
    raw_metrics = stability_metrics_from_series(
        batch_ids=kpi_df["Batch_ID"],
        values=kpi_df["AvgWeeklySales"],
        baseline_end=baseline_batches,
        tolerance_pct=tolerance_pct
    )
    stab_metrics = stability_metrics_from_series(
        batch_ids=kpi_df["Batch_ID"],
        values=kpi_df["AvgWeeklySales_Stabilized"],
        baseline_end=baseline_batches,
        tolerance_pct=tolerance_pct
    )

    table1 = pd.DataFrame([
        {"Scenario": "Raw (No Drift Detection)", **raw_metrics},
        {"Scenario": "With Drift Detection (Hold-Last-Stable)", **stab_metrics},
    ])

    # Alert log (Table2): full + alert-only
    drift_df["WarmupWindow"] = drift_df["Batch_ID"].between(warmup_start, warmup_end)
    drift_df["Threshold"] = threshold
    drift_df["ThresholdMethod"] = threshold_label

    table2_full = drift_df.copy()
    table2_alert = drift_df[drift_df["IsAlert"]].copy()

    # Save tables
    t1_path = str(Path(table_dir) / "RQ4_Table1.xlsx")
    t2_path = str(Path(table_dir) / "RQ4_Table2.xlsx")

    with pd.ExcelWriter(t1_path, engine="openpyxl") as w:
        table1.to_excel(w, index=False, sheet_name="Stability")

    with pd.ExcelWriter(t2_path, engine="openpyxl") as w:
        table2_alert.to_excel(w, index=False, sheet_name="Alert_Only")
        table2_full.to_excel(w, index=False, sheet_name="Full_Log")

    # Figure 1: KPI stability (raw vs stabilized)
    fig1_path = str(Path(fig_dir) / "RQ4_Fig1.pdf")
    plt.figure(figsize=(10, 4))
    plt.plot(kpi_df["Batch_ID"], kpi_df["AvgWeeklySales"], marker="o", linewidth=1, label="Raw AvgWeeklySales")
    plt.plot(kpi_df["Batch_ID"], kpi_df["AvgWeeklySales_Stabilized"], marker="o", linewidth=1, label="Stabilized AvgWeeklySales")
    plt.axvline(baseline_batches, linestyle="--", linewidth=1)
    plt.xlabel("Batch_ID (weekly)")
    plt.ylabel("Avg Weekly Sales")
    plt.title("RQ4_Fig1: Dashboard KPI Stability (Raw vs With Drift Detection)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig1_path)
    plt.close()

    # Figure 2: Drift trend with threshold + alerts
    fig2_path = str(Path(fig_dir) / "RQ4_Fig2.pdf")
    plt.figure(figsize=(10, 4))
    plt.plot(drift_df["Batch_ID"], drift_df["DriftScore_MedianPSI"], marker="o", linewidth=1, label="DriftScore (Median PSI)")
    plt.axhline(threshold, linestyle="--", linewidth=1, label=f"Threshold = {threshold:.3f}")
    alert_pts = drift_df[drift_df["IsAlert"]]
    if not alert_pts.empty:
        plt.scatter(alert_pts["Batch_ID"], alert_pts["DriftScore_MedianPSI"], marker="x", s=70, label="Alerts")
    plt.axvline(baseline_batches, linestyle="--", linewidth=1)
    plt.xlabel("Batch_ID (weekly)")
    plt.ylabel("Drift Score (Median PSI vs Baseline)")
    plt.title(f"RQ4_Fig2: Drift Trend and Alert Threshold ({threshold_label})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig2_path)
    plt.close()

    # Figure 3: measured improvement (example: within-tolerance %)
    fig3_path = str(Path(fig_dir) / "RQ4_Fig3.pdf")
    improvement_within = (stab_metrics["WithinTolerance_%"] - raw_metrics["WithinTolerance_%"]) if (
        stab_metrics["WithinTolerance_%"] == stab_metrics["WithinTolerance_%"]
        and raw_metrics["WithinTolerance_%"] == raw_metrics["WithinTolerance_%"]
    ) else np.nan

    fig3_df = pd.DataFrame({
        "Metric": ["WithinTolerance_% (post-baseline)"],
        "Raw": [raw_metrics["WithinTolerance_%"]],
        "WithDriftDetection": [stab_metrics["WithinTolerance_%"]],
        "Improvement": [improvement_within],
    })

    plt.figure(figsize=(8, 4))
    x = np.arange(len(fig3_df))
    plt.bar(x - 0.2, fig3_df["Raw"], width=0.4, label="Raw")
    plt.bar(x + 0.2, fig3_df["WithDriftDetection"], width=0.4, label="With Drift Detection")
    plt.xticks(x, fig3_df["Metric"], rotation=0)
    plt.ylabel("Percent (%)")
    plt.title("RQ4_Fig3: Measured Stability Improvement After Drift Detection (AvgWeeklySales)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig3_path)
    plt.close()

    # Alert rate
    alert_rate = float(drift_df["IsAlert"].mean() * 100.0)

    # Output
    print("RQ4 outputs generated successfully:")
    print(f"- {t1_path}")
    print(f"- {t2_path}")
    print(f"- {fig1_path}")
    print(f"- {fig2_path}")
    print(f"- {fig3_path}")
    print(f"Baseline batches used: 1..{baseline_batches}")
    print(f"Drift features used (count): {len(feature_cols)}")
    print(f"Threshold method: {threshold_label}")
    print(f"Threshold value: {threshold:.6f}")
    print(f"Alert rate across all batches: {alert_rate:.2f}%")
    print(f"Tolerance used for reliability (% from baseline mean): ±{tolerance_pct:.1f}%")


if __name__ == "__main__":
    generate_rq4_outputs(
        baseline_batches=10,
        warmup_batches=20,
        psi_bins=10,
        tolerance_pct=5.0,
        table_dir="tables",
        fig_dir="figures",
        threshold_method="WarmupQuantile",
        target_alert_rate=0.10,
    )
