"""
task_omnianomaly.py - Algorithm 13: OmniAnomaly (GRU + stochastic VAE).
DDP-trained and scored entirely from the contiguous ID-ordered shards -
only depends on task_write_seq_shards.py.
"""
import torch
import torch.nn as nn
from pyspark.sql import functions as F

from common import (
    Z_COLS, SEQ_SHARD_PATH, LSTM_WINDOW, N_FEATURES, OmniAnomalyLite,
    get_spark_session, distributed_train_sequential, score_sequential_distributed, save_algo_output,
)


def main():
    spark = get_spark_session("anomaly_omnianomaly")
    seq_shard_df = spark.read.parquet(SEQ_SHARD_PATH)
    id_type = seq_shard_df.schema["ID"].dataType

    def omni_loss(m, xb):
        xhat_last, mu, logvar = m(xb)
        recon = nn.functional.mse_loss(xhat_last, xb[:, -1, :])
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + 0.1 * kl

    omni_model = distributed_train_sequential(
        lambda: OmniAnomalyLite(N_FEATURES), omni_loss, SEQ_SHARD_PATH, Z_COLS, LSTM_WINDOW, label="OmniAnomaly"
    )

    def omni_score_fn(m, xb):
        xhat_last, mu, logvar = m(xb)
        recon = ((xhat_last - xb[:, -1, :]) ** 2).mean(dim=1)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return (recon + 0.1 * kl).numpy()

    omni_scores_df = score_sequential_distributed(
        spark, seq_shard_df, omni_model, lambda: OmniAnomalyLite(N_FEATURES),
        Z_COLS, LSTM_WINDOW, id_type, omni_score_fn, "omni_score"
    )

    stats = omni_scores_df.select(F.mean("omni_score").alias("m"), F.stddev_pop("omni_score").alias("s")).first()
    thr = stats["m"] + 0.6 * stats["s"]
    out = omni_scores_df.withColumn("omni_flag", F.when(F.col("omni_score") > F.lit(thr), 1).otherwise(0))

    save_algo_output(out.select("ID", "omni_score", "omni_flag"), "omnianomaly")
    spark.stop()


if __name__ == "__main__":
    main()