# Data Quality Drift Detection in Continuous ETL Pipelines



### Project Overview
This project implements an end-to-end data engineering pipeline to monitor and analyze data quality drift in continuous ETL systems using statistical techniques. The focus is on data reliability, distributional stability, and operational robustness rather than predictive machine learning.

The pipeline processes data in weekly batches, compares new data against a stable historical baseline, and detects drift using statistical measures. Apache Airflow is used to orchestrate ingestion, analysis, and reporting stages.

-----------------------------------------------------------------------------

### Dataset
Dataset: Walmart Sales Dataset
Source: Kaggle
Link: https://www.kaggle.com/datasets/yasserh/walmart-dataset

Only a sample CSV is included in this repository to ensure reproducibility while avoiding large raw data uploads.

-----------------------------------------------------------------------------

### Research Questions
RQ1: What is the baseline statistical profile of stable historical data batches?
RQ2: How does data quality (missing values, outliers, and distributions) change over time?
RQ3: What is the operational impact of detected data drift?
RQ4: How can alert thresholds be calibrated to balance sensitivity and false alarms?

Technical Approach
Weekly batch-based processing using Batch_ID
Statistical drift detection using PSI, KS, and deviation analysis
Data quality monitoring (missing values, outliers, variance)
Operational impact analysis
Threshold calibration and alert-rate evaluation
End-to-end pipeline orchestration with Apache Airflow

-----------------------------------------------------------------------------

### Project Structure
Drift_Detection_Walmart/
│
├── dags/
│ └── project_pipeline_dag.py
│
├── src/
│ ├── data_ingestion/
│ ├── data_cleaning/
│ ├── feature_engineering/
│ ├── modeling/
│ └── evaluation/
│
├── data/
│ └── sample/
│
├── figures/
├── tables/
│
├── requirements.txt
└── README.md

# Outputs
    All figures and tables are generated directly from code.
    No manual creation or editing is performed.
    Tables generated: 9 (.xlsx)
    Figures generated: 12 (.pdf)
    Total artifacts: 21
    Outputs are stored in the tables/ and figures/ directories.



Some subfolders under src/ (such as data_cleaning, feature_engineering, and modeling) are intentionally minimal or contain placeholder documentation. In this project, data cleaning, feature preparation, and modeling logic are embedded directly within the ingestion and evaluation stages, as the focus is on statistical data quality drift detection rather than standalone predictive modeling. The folder structure is retained to reflect a standard, extensible data engineering pipeline design and to allow future expansion without restructuring the repository.

-----------------------------------------------------------------------------

### How to Run Without Airflow
Install dependencies:
pip install -r requirements.txt

Run the pipeline scripts sequentially:
python -m src.data_ingestion.ingest_walmart
python -m src.evaluation.rq1_baseline_profiling
python -m src.evaluation.rq2_data_quality
python -m src.evaluation.rq3_impact_analysis
python -m src.evaluation.rq4_operational_impact

All tables and figures will be generated automatically.

-----------------------------------------------------------------------------

### How to Run Using Airflow (Local Setup)

# Prerequisites:
    Apache Airflow installed locally
    PostgreSQL running locally
    Database credentials configured in src/data_ingestion/db_engine.py
    Python environment with required packages installed

The pipeline reads from and writes to a local PostgreSQL database. Processed data is written during ingestion and read by downstream tasks.

# Make the DAG discoverable by Airflow using a symlink:
    ln -s <absolute-path-to-repo>/dags/project_pipeline_dag.py ~/airflow/dags/project_pipeline_dag.py

# Start Airflow services in two separate terminals:
Terminal 1:
    airflow scheduler

Terminal 2:
    airflow api-server --port 8080

# Access the Airflow UI in a browser:
http://localhost:8080

# Execute the DAG
DAG ID: project_pipeline_dag
Enable the DAG using the toggle
Trigger manually or allow scheduled execution

The DAG executes the full pipeline in order:
Data Ingestion -> RQ1 -> RQ2 -> RQ3 -> RQ4

Each task reads data from PostgreSQL and writes outputs to the tables/ and figures/ directories.

# Reproducibility
All results are generated programmatically
No manual table or figure creation
Clear and consistent file naming
Deterministic execution order enforced by the Airflow DAG
Database state is regenerated via ingestion

This ensures full reproducibility of all experiments and outputs.
