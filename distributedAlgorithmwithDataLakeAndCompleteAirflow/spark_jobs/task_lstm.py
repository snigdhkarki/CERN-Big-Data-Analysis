"""
task_lstm.py - Algorithm 12: LSTM sequence-autoencoder. DDP-trained and
scored entirely from the contiguous ID-ordered shards - no dependency
on task_prepare_features.py, only on task_write_seq_shards.py.
"""
import torch.nn as nn

from common import (
    Z_COLS, SEQ_SHARD_PATH, LSTM_WINDOW, N_FEATURES, LSTMAE,
    get_spark_session, distributed_train_sequential, score_sequential_distributed, save_algo_output,
)
from pyspark.sql import functions as F


def main():
    spark = get_spark_session("anomaly_lstm")
    seq_shard_df = spark.read.parquet(SEQ_SHARD_PATH)
    id_type = seq_shard_df.schema["ID"].dataType

    def lstm_loss(m, xb):
        xhat = m(xb)
        return nn.functional.mse_loss(xhat, xb)

    lstm_model = distributed_train_sequential(
        lambda: LSTMAE(N_FEATURES, LSTM_WINDOW), lstm_loss, SEQ_SHARD_PATH, Z_COLS, LSTM_WINDOW, label="LSTM"
    )

    def lstm_score_fn(m, xb):
        xhat = m(xb)
        return ((xhat[:, -1, :] - xb[:, -1, :]) ** 2).mean(dim=1).numpy()

    lstm_scores_df = score_sequential_distributed(
        spark, seq_shard_df, lstm_model, lambda: LSTMAE(N_FEATURES, LSTM_WINDOW),
        Z_COLS, LSTM_WINDOW, id_type, lstm_score_fn, "lstm_score"
    )

    stats = lstm_scores_df.select(F.mean("lstm_score").alias("m"), F.stddev_pop("lstm_score").alias("s")).first()
    thr = stats["m"] + 0.5 * stats["s"]
    out = lstm_scores_df.withColumn("lstm_flag", F.when(F.col("lstm_score") > F.lit(thr), 1).otherwise(0))

    save_algo_output(out.select("ID", "lstm_score", "lstm_flag"), "lstm")
    spark.stop()


if __name__ == "__main__":
    main()