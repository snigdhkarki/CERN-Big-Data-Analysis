"""
task_vae.py - Algorithm 8: Variational Autoencoder. Fully independent
DDP train + score, no shared state with any other algorithm task.
Depends on task_prepare_features.py and task_write_row_shards.py.
"""
import torch
import torch.nn as nn

from common import (
    Z_COLS, FEATURES_PATH, ROW_SHARD_PATH, TorchVAE,
    get_spark_session, distributed_train, make_torch_scalar_udf, threshold_flag, save_algo_output,
)


def main():
    spark = get_spark_session("anomaly_vae")
    df = spark.read.format("delta").load(FEATURES_PATH)

    def vae_loss(m, xb):
        xhat, mu, logvar = m(xb)
        recon = nn.functional.mse_loss(xhat, xb)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + 0.1 * kl

    vae = distributed_train(TorchVAE, vae_loss, ROW_SHARD_PATH, Z_COLS, label="VAE")

    def vae_score(m, xb):
        mu, logvar = m.encode(xb)
        xhat = m.decoder(mu)
        recon = ((xhat - xb) ** 2).mean(dim=1)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return (recon + 0.1 * kl).numpy()

    vae_udf = make_torch_scalar_udf(spark, TorchVAE, vae, vae_score, "vae_score")
    out = df.withColumn("vae_score", vae_udf(*Z_COLS))
    out = threshold_flag(out, "vae_score", "vae_flag", n_std=1.3)

    save_algo_output(out.select("ID", "vae_score", "vae_flag"), "vae")
    spark.stop()


if __name__ == "__main__":
    main()