"""
task_usad.py - Algorithm 10: USAD. Fully independent DDP train + score.
Depends on task_prepare_features.py and task_write_row_shards.py.
"""
import torch.nn as nn

from common import (
    Z_COLS, FEATURES_PATH, ROW_SHARD_PATH, USADModel,
    get_spark_session, distributed_train, make_torch_scalar_udf, threshold_flag, save_algo_output,
)


def main():
    spark = get_spark_session("anomaly_usad")
    df = spark.read.format("delta").load(FEATURES_PATH)

    def usad_loss(m, xb):
        w1, w2, w3 = m(xb)
        l_ae1 = nn.functional.mse_loss(w1, xb)
        l_ae2 = nn.functional.mse_loss(w2, xb)
        l_adv = nn.functional.mse_loss(w3, xb)
        return l_ae1 + l_ae2 - 0.5 * l_adv.detach() + 0.5 * l_adv

    usad = distributed_train(USADModel, usad_loss, ROW_SHARD_PATH, Z_COLS, label="USAD")

    def usad_score(m, xb):
        w1, _, w3 = m(xb)
        s1 = ((xb - w1) ** 2).mean(dim=1)
        s2 = ((xb - w3) ** 2).mean(dim=1)
        return (0.5 * s1 + 0.5 * s2).numpy()

    usad_udf = make_torch_scalar_udf(spark, USADModel, usad, usad_score, "usad_score")
    out = df.withColumn("usad_score", usad_udf(*Z_COLS))
    out = threshold_flag(out, "usad_score", "usad_flag", n_std=1.2)

    save_algo_output(out.select("ID", "usad_score", "usad_flag"), "usad")
    spark.stop()


if __name__ == "__main__":
    main()