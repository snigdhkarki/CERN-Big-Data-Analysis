#!/bin/bash
set -e

echo "Starting containers..."
sudo docker-compose up -d

echo "Waiting for Oracle..."
sleep 15

echo "Initializing Oracle table (drop + recreate)..."
cat init.sql | sudo docker-compose exec -T oracle sqlplus system/demo_password@localhost/XE

echo "Running ETL job..."
python etl_job.py

echo "Submitting anomaly detection to Spark..."
sudo docker-compose exec spark-master /opt/spark/bin/spark-submit \
  --master spark://localhost:7077 \
  /app/ml_anomaly.py

echo "Pipeline completed."