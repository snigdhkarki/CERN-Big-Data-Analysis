"""
network_traffic_pipeline_dag.py

Orchestrates the existing (already-working) shell pipeline with Airflow.
Airflow runs on the HOST (not in Docker), so these tasks call the exact
same commands run_pipeline.sh already calls. The docker-compose stack
(namenode, datanodes, oracle, spark-master, spark-workers) must already
be up before this DAG runs -- either start it yourself with
`docker-compose up -d`, or uncomment the optional `start_containers`
task below and set it upstream of task 1.

Set PROJECT_DIR below (or the AIRFLOW_VAR_PROJECT_DIR / Airflow Variable
"project_dir") to the absolute path containing docker-compose.yml,
init.sql, dcs_test.csv, and etl_job.py.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.operators.bash import BashOperator

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
# Prefer an Airflow Variable ("project_dir") so this isn't hardcoded;
# falls back to a literal path you should edit for your machine.
PROJECT_DIR = Variable.get("project_dir", default_var="/home/snigdh/repos/bigData")

default_args = {
    "owner": "you",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="network_traffic_pipeline",
    description="Load CSV -> Oracle, run ETL to HDFS/Delta, run Spark anomaly detection",
    default_args=default_args,
    schedule=None,          # trigger manually, or set e.g. "@daily"
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["spark", "oracle", "hdfs", "anomaly-detection"],
) as dag:

    # (Optional) Bring the stack up first. Uncomment and chain in front
    # of task 1 if you don't want to run `docker-compose up -d` by hand.
    #
    start_containers = BashOperator(
        task_id="start_containers",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "docker-compose up -d && "
            "sleep 15"   # give Oracle time to finish starting
        ),
    )

    # ------------------------------------------------------------------
    # Task 1: Load CSV into Oracle
    #   - copy dcs_test.csv into the oracle container
    #   - run init.sql (drops/recreates network_traffic, loads the CSV)
    # ------------------------------------------------------------------
    load_csv_to_oracle = BashOperator(
        task_id="load_csv_to_oracle",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "docker-compose exec -T oracle bash -c 'cat > /tmp/dcs_test.csv' < dcs_test.csv && "
            "cat init.sql | docker-compose exec -T oracle sqlplus system/demo_password@localhost/XE"
        ),
    )

    # ------------------------------------------------------------------
    # Task 2: Run ETL (Oracle -> HDFS as Delta table)
    #   Runs exactly like run_pipeline.sh: `python etl_job.py` on the
    #   host, using the local pyspark/delta-spark install and the
    #   localhost-published Oracle/HDFS ports.
    # ------------------------------------------------------------------
    run_etl = BashOperator(
        task_id="run_etl",
        bash_command=f"cd {PROJECT_DIR} && python etl_job.py",
    )

    # ------------------------------------------------------------------
    # Task 3: Run anomaly detection on the Spark cluster
    # ------------------------------------------------------------------
    run_anomaly_detection = BashOperator(
        task_id="run_anomaly_detection",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "docker-compose exec spark-master /opt/spark/bin/spark-submit "
            "--master spark://localhost:7077 "
            "--conf spark.ui.port=4040 "
            "--conf spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension "
            "--conf spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog "
            "/app/ml_anomaly.py"
        ),
    )

    start_containers >> load_csv_to_oracle >> run_etl >> run_anomaly_detection
