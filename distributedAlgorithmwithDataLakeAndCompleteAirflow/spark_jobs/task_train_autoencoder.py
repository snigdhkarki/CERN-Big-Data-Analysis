"""
task_train_autoencoder.py

Trains the single TorchAE that both Algorithm 6 (Deep Embedding) and
Algorithm 7 (DAGMM) build on top of, in the original script these two
literally shared the same in-memory model object. Now that they're
separate spark-submit processes, this task trains it once via DDP and
saves the state_dict to HDFS; task_deep_embedding.py and task_dagmm.py
each load it independently instead of retraining. Depends only on
task_write_row_shards.py.
"""
import torch.nn as nn

from common import Z_COLS, ROW_SHARD_PATH, AE_MODEL_PATH, TorchAE, distributed_train, save_torch_state, get_spark_session


def main():
    spark = get_spark_session("anomaly_train_autoencoder")

    def ae_loss(m, xb):
        _, xhat = m(xb)
        return nn.functional.mse_loss(xhat, xb)

    ae = distributed_train(TorchAE, ae_loss, ROW_SHARD_PATH, Z_COLS, label="AE")
    save_torch_state(spark, ae.state_dict(), AE_MODEL_PATH)

    spark.stop()


if __name__ == "__main__":
    main()