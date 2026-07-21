"""
task_write_seq_shards.py

Writes the contiguous, ID-ordered parquet shards LSTM and OmniAnomaly
train and score from. This is the one deliberately non-parallel step in
the whole pipeline (a single global Window.orderBy("ID")); isolating it
in its own task means a retry only re-runs that one global sort, not
the whole feature pipeline.
"""
from common import Z_COLS, FEATURES_PATH, SEQ_SHARD_PATH, NUM_WORKERS, get_spark_session, write_seq_shards


def main():
    spark = get_spark_session("anomaly_write_seq_shards")
    df = spark.read.format("delta").load(FEATURES_PATH)
    write_seq_shards(df, Z_COLS, SEQ_SHARD_PATH, NUM_WORKERS)
    spark.stop()


if __name__ == "__main__":
    main()