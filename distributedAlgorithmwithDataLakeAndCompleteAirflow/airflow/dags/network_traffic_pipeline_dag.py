"""
network_traffic_pipeline_dag.py

Orchestrates the pipeline with Airflow. Airflow runs on the HOST (not
in Docker); tasks call the same docker-compose/spark-submit commands
you'd run by hand. The docker-compose stack (namenode, datanodes,
oracle, spark-master, spark-workers) must already be up before this DAG
runs -- either start it yourself with `docker-compose up -d`, or leave
the `start_containers` task wired in as below.

Set PROJECT_DIR below (or the AIRFLOW_VAR_PROJECT_DIR / Airflow Variable
"project_dir") to the absolute path containing docker-compose.yml,
init.sql, dcs_test.csv, etl_job.py, and the spark_jobs/ directory.

WHAT CHANGED FROM THE ORIGINAL SINGLE-TASK VERSION
====================================================
The old `run_anomaly_detection` task ran one `spark-submit` of a
1,000-line ml_anomaly.py covering all 14 anomaly-detection algorithms.
A crash in, say, algorithm #9 (GAN) meant retrying the whole thing,
including everything before it that had already succeeded.

ml_anomaly.py is now split into spark_jobs/task_*.py (see
spark_jobs/README.md for the full data-flow diagram). Each algorithm is
its own spark-submit call and its own Airflow task, so a failure only
costs a retry of that one task:

  prepare_features
  -> write_row_shards -> {vae, gan, usad, transformer, train_autoencoder -> {deep_embedding, dagmm}}
  -> write_seq_shards -> {lstm, omnianomaly}
  -> {kmeans, gmm, pca_kmeans, dbscan, isolation_forest, rl_adaptive_threshold}
  -> ensemble_report (fans in on all 14 algorithm outputs)

Every spark-submit call includes `--py-files /app/common.py` -- the
model classes and DDP/UDF helpers used to live in ml_anomaly.py's own
__main__ (which Spark ships to executors "by value" automatically);
now that they live in a separate common.py module, executors need
--py-files to resolve `import common`. See spark_jobs/README.md.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.operators.bash import BashOperator
from airflow.utils.task_group import TaskGroup

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
PROJECT_DIR = Variable.get("project_dir", default_var="/home/snigdh/repos/bigData")

SPARK_SUBMIT_CONF = (
    "--conf spark.ui.port=4040 "
    "--conf spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension "
    "--conf spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog"
)


def spark_submit_cmd(script_name: str) -> str:
    """One-liner for `docker-compose exec spark-master spark-submit ... /app/<script_name>`.
    Every algorithm task shares this same shape; --py-files ships
    common.py to executors so `import common` works there too."""
    return (
        f"cd {PROJECT_DIR} && "
        "docker-compose exec spark-master /opt/spark/bin/spark-submit "
        "--master spark://localhost:7077 "
        "--py-files /app/common.py "
        f"{SPARK_SUBMIT_CONF} "
        f"/app/{script_name}"
    )


default_args = {
    "owner": "you",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

# DDP-training tasks are the ones most likely to need a second attempt
# (executor scheduling races, transient OOM) -- give them one extra
# retry and a slightly longer backoff than the cheap Spark-native tasks.
DDP_TRAINING_ARGS = {
    **default_args,
    "retries": 2,
    "retry_delay": timedelta(minutes=3),
}

with DAG(
    dag_id="network_traffic_pipeline",
    description="Load CSV -> Oracle, run ETL to HDFS/Delta, run per-algorithm anomaly detection on Spark",
    default_args=default_args,
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["spark", "oracle", "hdfs", "anomaly-detection"],
) as dag:

    start_containers = BashOperator(
        task_id="start_containers",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "docker-compose up -d && "
            "sleep 15"
        ),
    )

    # ------------------------------------------------------------------
    # Load CSV into Oracle, then Oracle -> HDFS/Delta via etl_job.py
    # (unchanged from the original DAG)
    # ------------------------------------------------------------------
    load_csv_to_oracle = BashOperator(
        task_id="load_csv_to_oracle",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "docker-compose exec -T oracle bash -c 'cat > /tmp/dcs_test.csv' < dcs_test.csv && "
            "cat init.sql | docker-compose exec -T oracle sqlplus system/demo_password@localhost/XE"
        ),
    )

    run_etl = BashOperator(
        task_id="run_etl",
        bash_command=f"cd {PROJECT_DIR} && python etl_job.py",
    )

    # ------------------------------------------------------------------
    # Feature prep - one shared StandardScaler fit, reused by every
    # algorithm below instead of each one recomputing it.
    # ------------------------------------------------------------------
    prepare_features = BashOperator(
        task_id="prepare_features",
        bash_command=spark_submit_cmd("task_prepare_features.py"),
    )

    # ------------------------------------------------------------------
    # Shard writers - one for the i.i.d.-row models, one for the
    # contiguous ID-ordered sequence models.
    # ------------------------------------------------------------------
    with TaskGroup(group_id="write_shards") as write_shards:
        write_row_shards = BashOperator(
            task_id="write_row_shards",
            bash_command=spark_submit_cmd("task_write_row_shards.py"),
        )
        write_seq_shards = BashOperator(
            task_id="write_seq_shards",
            bash_command=spark_submit_cmd("task_write_seq_shards.py"),
        )

    # ------------------------------------------------------------------
    # Algorithms 1-3: native pyspark.ml, always fully distributed.
    # Only need prepare_features.
    # ------------------------------------------------------------------
    with TaskGroup(group_id="native_clustering") as native_clustering:
        algo_kmeans = BashOperator(
            task_id="kmeans",
            bash_command=spark_submit_cmd("task_kmeans.py"),
        )
        algo_gmm = BashOperator(
            task_id="gmm",
            bash_command=spark_submit_cmd("task_gmm.py"),
        )
        algo_pca_kmeans = BashOperator(
            task_id="pca_kmeans",
            bash_command=spark_submit_cmd("task_pca_kmeans.py"),
        )

    # ------------------------------------------------------------------
    # Algorithms 4-5: parallel per-partition sklearn fit via mapInPandas.
    # Only need prepare_features.
    # ------------------------------------------------------------------
    with TaskGroup(group_id="partition_local_models") as partition_local_models:
        algo_dbscan = BashOperator(
            task_id="dbscan",
            bash_command=spark_submit_cmd("task_dbscan.py"),
        )
        algo_isolation_forest = BashOperator(
            task_id="isolation_forest",
            bash_command=spark_submit_cmd("task_isolation_forest.py"),
        )

    # ------------------------------------------------------------------
    # Algorithms 6-7: share one trained TorchAE. Train once, then fan
    # out to Deep Embedding and DAGMM so neither retrains it.
    # ------------------------------------------------------------------
    with TaskGroup(group_id="autoencoder_family") as autoencoder_family:
        train_autoencoder = BashOperator(
            task_id="train_autoencoder",
            bash_command=spark_submit_cmd("task_train_autoencoder.py"),
            **DDP_TRAINING_ARGS,
        )
        algo_deep_embedding = BashOperator(
            task_id="deep_embedding",
            bash_command=spark_submit_cmd("task_deep_embedding.py"),
        )
        algo_dagmm = BashOperator(
            task_id="dagmm",
            bash_command=spark_submit_cmd("task_dagmm.py"),
        )
        train_autoencoder >> [algo_deep_embedding, algo_dagmm]

    # ------------------------------------------------------------------
    # Algorithms 8-11: fully independent row-wise DDP models.
    # ------------------------------------------------------------------
    with TaskGroup(group_id="row_wise_ddp_models") as row_wise_ddp_models:
        algo_vae = BashOperator(
            task_id="vae",
            bash_command=spark_submit_cmd("task_vae.py"),
            **DDP_TRAINING_ARGS,
        )
        algo_gan = BashOperator(
            task_id="gan",
            bash_command=spark_submit_cmd("task_gan.py"),
            **DDP_TRAINING_ARGS,
        )
        algo_usad = BashOperator(
            task_id="usad",
            bash_command=spark_submit_cmd("task_usad.py"),
            **DDP_TRAINING_ARGS,
        )
        algo_transformer = BashOperator(
            task_id="transformer",
            bash_command=spark_submit_cmd("task_transformer.py"),
            **DDP_TRAINING_ARGS,
        )

    # ------------------------------------------------------------------
    # Algorithms 12-13: sequential DDP models over the contiguous shards.
    # ------------------------------------------------------------------
    with TaskGroup(group_id="sequential_ddp_models") as sequential_ddp_models:
        algo_lstm = BashOperator(
            task_id="lstm",
            bash_command=spark_submit_cmd("task_lstm.py"),
            **DDP_TRAINING_ARGS,
        )
        algo_omnianomaly = BashOperator(
            task_id="omnianomaly",
            bash_command=spark_submit_cmd("task_omnianomaly.py"),
            **DDP_TRAINING_ARGS,
        )

    # ------------------------------------------------------------------
    # Algorithm 14: RL-tuned adaptive threshold. Cheap, only needs
    # prepare_features.
    # ------------------------------------------------------------------
    algo_rl_adaptive_threshold = BashOperator(
        task_id="rl_adaptive_threshold",
        bash_command=spark_submit_cmd("task_rl_adaptive_threshold.py"),
    )

    # ------------------------------------------------------------------
    # Fan-in: join all 14 algorithm outputs, compute the ensemble vote,
    # print the report, persist the final table.
    # ------------------------------------------------------------------
    ensemble_report = BashOperator(
        task_id="ensemble_report",
        bash_command=spark_submit_cmd("task_ensemble_report.py"),
    )

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------
    start_containers >> load_csv_to_oracle >> run_etl >> prepare_features

    prepare_features >> write_shards
    prepare_features >> native_clustering
    prepare_features >> partition_local_models
    prepare_features >> algo_rl_adaptive_threshold

    write_row_shards >> [algo_vae, algo_gan, algo_usad, algo_transformer, train_autoencoder]
    write_seq_shards >> [algo_lstm, algo_omnianomaly]

    all_algorithms = [
        algo_kmeans, algo_gmm, algo_pca_kmeans,
        algo_dbscan, algo_isolation_forest,
        algo_deep_embedding, algo_dagmm,
        algo_vae, algo_gan, algo_usad, algo_transformer,
        algo_lstm, algo_omnianomaly,
        algo_rl_adaptive_threshold,
    ]
    all_algorithms >> ensemble_report