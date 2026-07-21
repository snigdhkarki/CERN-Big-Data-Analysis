"""
task_transformer.py - Algorithm 11: Transformer-based reconstruction.
Fully independent DDP train + score. Depends on task_prepare_features.py
and task_write_row_shards.py.
"""
import torch.nn as nn

from common import (
    Z_COLS, FEATURES_PATH, ROW_SHARD_PATH, TinyTransformerAE,
    get_spark_session, distributed_train, make_torch_scalar_udf, threshold_flag, save_algo_output,
)


def main():
    spark = get_spark_session("anomaly_transformer")
    df = spark.read.format("delta").load(FEATURES_PATH)

    def trf_loss(m, xb):
        return nn.functional.mse_loss(m(xb), xb)

    trf = distributed_train(TinyTransformerAE, trf_loss, ROW_SHARD_PATH, Z_COLS, label="Transformer")

    def trf_score(m, xb):
        return ((m(xb) - xb) ** 2).mean(dim=1).numpy()

    trf_udf = make_torch_scalar_udf(spark, TinyTransformerAE, trf, trf_score, "transformer_score")
    out = df.withColumn("transformer_score", trf_udf(*Z_COLS))
    out = threshold_flag(out, "transformer_score", "transformer_flag", n_std=0.2)

    save_algo_output(out.select("ID", "transformer_score", "transformer_flag"), "transformer")
    spark.stop()


if __name__ == "__main__":
    main()