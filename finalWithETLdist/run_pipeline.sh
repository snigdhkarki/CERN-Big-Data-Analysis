#!/bin/bash
set -e

echo "Starting containers..."
sudo docker-compose up -d

echo "Waiting for Oracle..."
sleep 15

echo "Copying CSV into Oracle container..."
sudo docker-compose exec -T oracle bash -c 'cat > /tmp/dcs_test.csv' < dcs_test.csv

echo "Initializing Oracle table (drop + recreate)..."
cat init.sql | sudo docker-compose exec -T oracle sqlplus system/demo_password@localhost/XE

echo "Running ETL job..."
sudo docker-compose exec spark-master /opt/spark/bin/spark-submit \
  --master spark://localhost:7077 \
  --jars /app/ojdbc8.jar \
  --conf spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension \
  --conf spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog \
  /app/etl_job.py

echo "Running anomaly detection (split into per-algorithm Spark jobs -- see spark_jobs/README.md)..."
./spark_jobs/run_all_local.sh

echo "Pipeline completed."