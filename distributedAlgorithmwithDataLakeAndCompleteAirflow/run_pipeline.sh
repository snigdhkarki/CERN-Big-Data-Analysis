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
python etl_job.py

echo "Running anomaly detection (split into per-algorithm Spark jobs -- see spark_jobs/README.md)..."
./spark_jobs/run_all_local.sh

echo "Pipeline completed."