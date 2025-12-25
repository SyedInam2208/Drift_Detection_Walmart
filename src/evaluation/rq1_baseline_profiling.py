import pandas as pd
import numpy as np
from pathlib import Path


NUMERIC_COLS_DEFAULT = ["Weekly_Sales", "Temperature", "Fuel_Price", "CPI", "Unemployment"]
CATEGORICAL_COLS_DEFAULT = ["Store", "Holiday_Flag"]


def _missing_value_percent(df: pd.DataFrame) -> float:
    # % of missing cells across the whole dataframe
    total_cells = df.shape[0] * df.shape[1]
    if total_cells == 0:
        return 0.0
    return float(df.isna().sum().sum() / total_cells * 100)


def _duplicate_row_count(df: pd.DataFrame) -> int:
    return int(df.duplicated().sum())


def _incorrect_datatype_count(df: pd.DataFrame, expected_types: dict) -> int:
    """
    expected_types example:
    {
      "Date": "datetime64[ns]",
      "Store": "int64",
      ...
    }
    We count mismatches for columns present in df.
    """
    mismatches = 0
    for col, exp_type in expected_types.items():
        if col not in df.columns:
            continue
        actual = str(df[col].dtype)
        if actual != exp_type:
            mismatches += 1
    return int(mismatches)


def _outlier_percent_iqr(df: pd.DataFrame, numeric_cols: list) -> float:
    """
    Simple outlier percentage using IQR rule aggregated across numeric columns:
    outlier if value < Q1 - 1.5*IQR or > Q3 + 1.5*IQR
    We count outlier flags per numeric column and average.
    """
    if df.empty:
        return 0.0

    outlier_rates = []
    for col in numeric_cols:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            outlier_rates.append(0.0)
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_flags = (series < lower) | (series > upper)
        outlier_rates.append(outlier_flags.mean() * 100)

    if not outlier_rates:
        return 0.0
    return float(np.mean(outlier_rates))


def _unique_category_count(df: pd.DataFrame, col: str) -> int:
    if col not in df.columns:
        return 0
    return int(df[col].nunique(dropna=True))


def build_rq1_tables(
    processed_csv_path: str = "data/sample/walmart_processed.csv",
    baseline_batches: int = 10,
    new_batch_id: int | None = None,
    numeric_cols: list | None = None,
    categorical_cols: list | None = None,
    output_dir: str = "tables"
):
    """
    Generates:
      - tables/RQ1_Table1.xlsx : Baseline vs New Batch Quality Metrics
      - tables/RQ1_Table2.xlsx : Summary of Drift Types Detected (rule-based)
    """

    numeric_cols = numeric_cols or NUMERIC_COLS_DEFAULT
    categorical_cols = categorical_cols or CATEGORICAL_COLS_DEFAULT

    df = pd.read_csv(processed_csv_path)

    # Safety checks
    if "Batch_ID" not in df.columns:
        raise ValueError("Batch_ID not found. Run ingestion step to create walmart_processed.csv first.")

    # Convert Date if present (robust)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Baseline selection
    baseline_df = df[df["Batch_ID"] <= baseline_batches].copy()

    # Pick a "new batch" to compare:
    # If user didn't provide, use the last batch (most recent week)
    if new_batch_id is None:
        new_batch_id = int(df["Batch_ID"].max())
    new_batch_df = df[df["Batch_ID"] == new_batch_id].copy()

    if baseline_df.empty:
        raise ValueError("Baseline dataframe is empty. Increase baseline_batches or verify Batch_ID values.")
    if new_batch_df.empty:
        raise ValueError(f"New batch dataframe is empty for Batch_ID={new_batch_id}. Choose an existing batch.")

    # Expected dtype mapping (only those that exist will be checked)
    expected_types = {
        "Date": "datetime64[ns]",
        "Store": "int64",
        "Holiday_Flag": "int64",
        "Weekly_Sales": "float64",
        "Temperature": "float64",
        "Fuel_Price": "float64",
        "CPI": "float64",
        "Unemployment": "float64",
        "Batch_ID": "int64",
    }

    # Compute quality metrics: baseline vs new
    baseline_missing = _missing_value_percent(baseline_df)
    new_missing = _missing_value_percent(new_batch_df)

    baseline_outlier = _outlier_percent_iqr(baseline_df, numeric_cols)
    new_outlier = _outlier_percent_iqr(new_batch_df, numeric_cols)

    baseline_dupes = _duplicate_row_count(baseline_df)
    new_dupes = _duplicate_row_count(new_batch_df)

    baseline_dtype_mismatch = _incorrect_datatype_count(baseline_df, expected_types)
    new_dtype_mismatch = _incorrect_datatype_count(new_batch_df, expected_types)

    baseline_unique_store = _unique_category_count(baseline_df, "Store")
    new_unique_store = _unique_category_count(new_batch_df, "Store")

    table1 = pd.DataFrame({
        "Data Quality Metric": [
            "Missing Value % (overall)",
            "Outlier % (IQR rule, avg across numeric cols)",
            "Incorrect Data Types (column dtype mismatches)",
            "Duplicate Records (row duplicates)",
            "Unique Categories (Store)"
        ],
        "Baseline Value": [
            round(baseline_missing, 3),
            round(baseline_outlier, 3),
            baseline_dtype_mismatch,
            baseline_dupes,
            baseline_unique_store
        ],
        "New Batch Value": [
            round(new_missing, 3),
            round(new_outlier, 3),
            new_dtype_mismatch,
            new_dupes,
            new_unique_store
        ]
    })

    # ---- Rule-based drift summary for Table 1.2 ----
    drift_rows = []

    # Missing-value drift
    if new_missing > baseline_missing * 1.5 and (new_missing - baseline_missing) > 0.1:
        drift_rows.append(("Missing-value Drift", "Increase in nulls", "Multiple columns", "Cleaning fills nulls but does not monitor trend"))
    else:
        drift_rows.append(("Missing-value Drift", "Stable / small change", "Multiple columns", "No strong missingness drift signal"))

    # Distribution drift proxy using outliers
    if new_outlier > baseline_outlier * 1.5 and (new_outlier - baseline_outlier) > 0.1:
        drift_rows.append(("Distribution / Anomaly Drift", "Outlier rate spike (IQR)", "Numeric features", "Outlier removal may hide drift rather than detect it"))
    else:
        drift_rows.append(("Distribution / Anomaly Drift", "Stable outlier rate", "Numeric features", "No strong anomaly drift signal"))

    # Category drift: new stores appearing in new batch vs baseline
    if new_unique_store > baseline_unique_store:
        drift_rows.append(("Category Drift", "New category values appear", "Store", "Encoding masks frequency change without alerting"))
    else:
        drift_rows.append(("Category Drift", "No new categories", "Store", "No category expansion detected"))

    # Schema/type drift proxy
    if new_dtype_mismatch > 0:
        drift_rows.append(("Schema / Type Drift", "Unexpected dtype mismatch", "Schema check", "Cleaning may coerce types silently without alerting"))
    else:
        drift_rows.append(("Schema / Type Drift", "No dtype mismatch", "Schema check", "No schema drift detected"))

    table2 = pd.DataFrame(drift_rows, columns=[
        "Drift Type",
        "Description",
        "Example Column",
        "Why Traditional Cleaning Fails / Notes"
    ])

    # ---- Save outputs ----
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Strict naming (MANDATORY)
    table1_path = out_dir / "RQ1_Table1.xlsx"
    table2_path = out_dir / "RQ1_Table2.xlsx"

    with pd.ExcelWriter(table1_path, engine="openpyxl") as writer:
        table1.to_excel(writer, index=False, sheet_name="RQ1_Table1")

    with pd.ExcelWriter(table2_path, engine="openpyxl") as writer:
        table2.to_excel(writer, index=False, sheet_name="RQ1_Table2")

    print("RQ1 tables generated successfully:")
    print(f"- {table1_path}")
    print(f"- {table2_path}")
    print(f"Baseline batches used: 1..{baseline_batches}")
    print(f"New batch compared: Batch_ID={new_batch_id}")

    return table1, table2


if __name__ == "__main__":
    build_rq1_tables()

