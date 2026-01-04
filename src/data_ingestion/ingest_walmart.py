import pandas as pd
from pathlib import Path


def ingest_walmart_data(
    input_csv_path: str,
    output_csv_path: str,
    table_name: str = "walmart_processed",
    if_exists: str = "replace",
) -> pd.DataFrame:
    """
    Ingest Walmart dataset, clean dates, create weekly batches,
    save processed CSV, and load processed data into PostgreSQL.

    Parameters
    ----------
    input_csv_path : str
        Path to raw Walmart CSV (e.g., data/sample/Walmart.csv)
    output_csv_path : str
        Path to write processed CSV (e.g., data/sample/walmart_processed.csv)
    table_name : str
        PostgreSQL table to write (default: walmart_processed)
    if_exists : str
        Behavior if table exists: 'replace' or 'append' (default: replace)
    """

    # -----------------------------
    # Load raw data
    # -----------------------------
    df = pd.read_csv(input_csv_path)

    # -----------------------------
    # Basic cleaning + feature engineering
    # -----------------------------
    # Convert Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors="coerce")

    # Drop rows with invalid dates (defensive)
    df = df.dropna(subset=["Date"]).copy()

    # Sort chronologically
    df = df.sort_values("Date").reset_index(drop=True)

    # Create Year-Week identifier
    df["Year_Week"] = df["Date"].dt.strftime("%Y-%U")

    # Create numeric Batch_ID (weekly batches)
    df["Batch_ID"] = df["Year_Week"].factorize()[0] + 1

    # Enforce stable dtypes (important for DB + downstream)
    df["Batch_ID"] = df["Batch_ID"].astype(int)

    # Optional: ensure key numeric columns are numeric if present
    for col in ["Weekly_Sales", "Temperature", "Fuel_Price", "CPI", "Unemployment"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Optional: ensure IsHoliday is boolean-ish if present
    if "IsHoliday" in df.columns:
        # Handles True/False, 0/1, "TRUE"/"FALSE"
        df["IsHoliday"] = df["IsHoliday"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])

    # -----------------------------
    # Save processed CSV
    # -----------------------------
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)

    # -----------------------------
    # Write processed data to PostgreSQL (source of truth)
    # -----------------------------
    # IMPORTANT: correct import path
    from src.data_ingestion.db_engine import get_engine  # noqa: E402

    engine = get_engine()

    df.to_sql(
        name=table_name,
        con=engine,
        if_exists=if_exists,
        index=False,
        chunksize=5000,  # safer for larger data
        method="multi",
    )

    print("Ingestion completed successfully.")
    print(f"Input file: {input_csv_path}")
    print(f"Processed CSV: {output_csv_path}")
    print(f"DB table: {table_name} (if_exists={if_exists})")
    print(f"Total rows: {df.shape[0]}")
    print(f"Total weekly batches: {df['Batch_ID'].nunique()}")

    return df


if __name__ == "__main__":
    ingest_walmart_data(
        input_csv_path="data/sample/Walmart.csv",
        output_csv_path="data/sample/walmart_processed.csv",
        table_name="walmart_processed",
        if_exists="replace",
    )
