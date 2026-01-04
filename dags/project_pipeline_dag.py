from __future__ import annotations

import pendulum
from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator

# Timezone for scheduling (Berlin)
local_tz = pendulum.timezone("Europe/Berlin")

# Absolute path to your project repo root
PROJECT_ROOT = "/Users/syedinam/Desktop/UE_Lectures/DataEngineering/Project/Github_Clones/Drift_Detection_Walmart"

default_args = {
    "owner": "de_project",
    "retries": 0,
}

with DAG(
    dag_id="project_pipeline_dag",
    default_args=default_args,
    description="Walmart Drift Detection: ingestion -> RQ1..RQ4 (tables/figures generated from code)",
    # IMPORTANT: Use pendulum.datetime (supports tz=...)
    start_date=pendulum.datetime(2025, 12, 1, 6, 0, tz=local_tz),
    # Daily at 06:00 Berlin time
    schedule="0 6 * * *",
    catchup=False,
    max_active_runs=1,
    tags=["data-engineering", "walmart", "drift"],
) as dag:

    t_ingest = BashOperator(
        task_id="ingest_and_load_to_postgres",
        bash_command=f"cd {PROJECT_ROOT} && python -m src.data_ingestion.ingest_walmart",
    )

    t_rq1_profile = BashOperator(
        task_id="rq1_baseline_profiling",
        bash_command=f"cd {PROJECT_ROOT} && python -m src.evaluation.rq1_baseline_profiling",
    )

    t_rq2 = BashOperator(
        task_id="rq2_data_quality_tests",
        bash_command=f"cd {PROJECT_ROOT} && python -m src.evaluation.rq2_data_quality",
    )

    t_rq3 = BashOperator(
        task_id="rq3_impact_analysis",
        bash_command=f"cd {PROJECT_ROOT} && python -m src.evaluation.rq3_impact_analysis",
    )

    t_rq4 = BashOperator(
        task_id="rq4_operational_impact",
        bash_command=f"cd {PROJECT_ROOT} && python -m src.evaluation.rq4_operational_impact",
    )

    # End-to-end dependency chain
    t_ingest >> t_rq1_profile >> t_rq2 >> t_rq3 >> t_rq4
