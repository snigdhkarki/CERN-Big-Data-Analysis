"""
task_gan.py - Algorithm 9: GAN (Generator/Discriminator). Fully
independent DDP train + score; only the trained Discriminator is used
for scoring. Depends on task_prepare_features.py and
task_write_row_shards.py.
"""
from common import (
    Z_COLS, FEATURES_PATH, ROW_SHARD_PATH, Generator, Discriminator,
    get_spark_session, distributed_train_gan, make_torch_scalar_udf, threshold_flag, save_algo_output,
)


def main():
    spark = get_spark_session("anomaly_gan")
    df = spark.read.format("delta").load(FEATURES_PATH)

    noise_dim = 8
    _, D = distributed_train_gan(lambda: Generator(noise_dim), Discriminator, ROW_SHARD_PATH, Z_COLS, noise_dim,
                                  epochs=30, lr=5e-5)

    def gan_score(m, xb):
        return (1.0 - m(xb)).squeeze(-1).numpy()

    gan_udf = make_torch_scalar_udf(spark, Discriminator, D, gan_score, "gan_score")
    out = df.withColumn("gan_score", gan_udf(*Z_COLS))
    out = threshold_flag(out, "gan_score", "gan_flag", n_std=2.5)

    save_algo_output(out.select("ID", "gan_score", "gan_flag"), "gan")
    spark.stop()


if __name__ == "__main__":
    main()