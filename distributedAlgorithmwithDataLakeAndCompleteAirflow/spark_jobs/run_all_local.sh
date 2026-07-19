#!/bin/bash
# Runs every split-out task in dependency order, sequentially, via
# `docker-compose exec spark-master spark-submit`. This is NOT what
# Airflow uses (see network_traffic_pipeline_dag.py for the real,
# parallel, independently-retryable task graph) - it's just a quick way
# to smoke-test the whole chain by hand from the project root, the same
# way the old run_pipeline.sh smoke-tested the monolithic ml_anomaly.py.
set -e

SPARK_CONF="--conf spark.ui.port=4040 \
  --conf spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension \
  --conf spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog"

submit() {
  echo -e "\n=== $1 ==="
  sudo docker-compose exec spark-master /opt/spark/bin/spark-submit \
    --master spark://localhost:7077 \
    --py-files /app/common.py \
    $SPARK_CONF \
    "/app/$1"
}

submit task_prepare_features.py

submit task_write_row_shards.py
submit task_write_seq_shards.py

submit task_kmeans.py
submit task_gmm.py
submit task_pca_kmeans.py
submit task_dbscan.py
submit task_isolation_forest.py

submit task_train_autoencoder.py
submit task_deep_embedding.py
submit task_dagmm.py

submit task_vae.py
submit task_gan.py
submit task_usad.py
submit task_transformer.py

submit task_lstm.py
submit task_omnianomaly.py

submit task_rl_adaptive_threshold.py

submit task_ensemble_report.py

echo -e "\nPipeline completed."