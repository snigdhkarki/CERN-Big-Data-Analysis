"""
task_write_row_shards.py

Writes the randomly-sharded parquet each row-wise DDP model (AE, VAE,
GAN, USAD, Transformer) trains from. Split out as its own task so a
transient HDFS write failure doesn't force re-running the StandardScaler
fit in task_prepare_features.py.
"""
from common import Z_COLS, FEATURES_PATH, ROW_SHARD_PATH, NUM_WORKERS, get_spark_session, write_row_shards


def main():
    spark = get_spark_session("anomaly_write_row_shards")
    df = spark.read.format("delta").load(FEATURES_PATH)
    write_row_shards(df, Z_COLS, ROW_SHARD_PATH, NUM_WORKERS)
    spark.stop()


if __name__ == "__main__":
    main()