"""
task_deep_embedding.py - Algorithm 6: Deep Embedding (Autoencoder +
KMeans). Loads the AE trained by task_train_autoencoder.py rather than
retraining it. Depends on task_prepare_features.py and
task_train_autoencoder.py.
"""
from pyspark.ml.clustering import KMeans
from pyspark.ml.functions import array_to_vector

from common import (
    Z_COLS, FEATURES_PATH, AE_MODEL_PATH, RANDOM_SEED, LATENT, TorchAE,
    get_spark_session, load_torch_state, make_torch_array_udf, minority_cluster_flag, save_algo_output,
)


def main():
    spark = get_spark_session("anomaly_deep_embedding")
    df = spark.read.format("delta").load(FEATURES_PATH)

    ae = TorchAE()
    ae.load_state_dict(load_torch_state(spark, AE_MODEL_PATH))
    ae.eval()

    def embed_latent(m, xb):
        z, _ = m(xb)
        return z.numpy()

    embed_udf = make_torch_array_udf(spark, TorchAE, ae, embed_latent, LATENT, "deep_embed")
    out = df.withColumn("deepemb_arr", embed_udf(*Z_COLS))
    out = out.withColumn("deepemb_vec", array_to_vector("deepemb_arr"))

    deepemb_kmeans = KMeans(featuresCol="deepemb_vec", predictionCol="deepemb_cluster", k=8, seed=RANDOM_SEED)
    out = deepemb_kmeans.fit(out).transform(out)
    out = minority_cluster_flag(out, "deepemb_cluster", "deepemb_flag")

    save_algo_output(out.select("ID", "deepemb_cluster", "deepemb_flag"), "deep_embedding")
    spark.stop()


if __name__ == "__main__":
    main()