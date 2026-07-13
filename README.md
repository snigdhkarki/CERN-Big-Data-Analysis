# Forcefully remove the specific container causing the block
sudo docker rm -f bigdata_spark-master_1

# Alternatively, tear down the whole local stack structure safely
sudo docker-compose down

# Make your custom image
docker build -t my-spark:3.5.0-numpy -f docker.spark .

# Run the pipeline
sudo ./run_pipeline.sh