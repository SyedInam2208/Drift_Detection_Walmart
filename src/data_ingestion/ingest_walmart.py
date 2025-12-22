import pandas as pd
from pathlib import Path

def ingest_walmart_data(
    input_csv_path: str,
    output_csv_path: str
):
    """
    Ingest Walmart dataset, clean dates, create weekly batches,
    and save processed data for downstream tasks.
    """

    # Load raw data
    df = pd.read_csv(input_csv_path)

    # Convert Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")

    # Sort chronologically
    df = df.sort_values("Date").reset_index(drop=True)

    # Create Year-Week identifier
    df["Year_Week"] = df["Date"].dt.strftime("%Y-%U")

    # Create numeric Batch_ID (weekly batches)
    df["Batch_ID"] = df["Year_Week"].factorize()[0] + 1

    # Ensure output directory exists
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save processed dataset
    df.to_csv(output_csv_path, index=False)

    print("Ingestion completed successfully.")
    print(f"Total rows: {df.shape[0]}")
    print(f"Total weekly batches: {df['Batch_ID'].nunique()}")

    return df


if __name__ == "__main__":
    ingest_walmart_data(
        input_csv_path="data/sample/Walmart.csv",
        output_csv_path="data/sample/walmart_processed.csv"
    )
