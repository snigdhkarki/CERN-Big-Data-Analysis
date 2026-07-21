"""
task_dagmm.py - Algorithm 7: DAGMM. Loads the same AE that
task_deep_embedding.py loads (trained once by task_train_autoencoder.py)
and fits its own GaussianMixture on the reconstruction-error representation.
Depends on task_prepare_features.py and task_train_autoencoder.py.
"""
import torch
import torch.nn as nn
from pyspark.sql import functions as F
from pyspark.ml.clustering import GaussianMixture
from pyspark.ml.functions import array_to_vector, vector_to_array

from common import (
    Z_COLS, FEATURES_PATH, AE_MODEL_PATH, LATENT, RANDOM_SEED, TorchAE,
    get_spark_session, load_torch_state, make_torch_array_udf, threshold_flag, save_algo_output,
)


def main():
    spark = get_spark_session("anomaly_dagmm")
    df = spark.read.format("delta").load(FEATURES_PATH)

    ae = TorchAE()
    ae.load_state_dict(load_torch_state(spark, AE_MODEL_PATH))
    ae.eval()

    def dagmm_repr(m, xb):
        z, xhat = m(xb)
        euclid = torch.norm(xb - xhat, dim=1, keepdim=True)
        cos = nn.functional.cosine_similarity(xb, xhat, dim=1).unsqueeze(1)
        return torch.cat([z, euclid, cos], dim=1).numpy()

    dagmm_udf = make_torch_array_udf(spark, TorchAE, ae, dagmm_repr, LATENT + 2, "dagmm_repr")
    out = df.withColumn("dagmm_arr", dagmm_udf(*Z_COLS))
    out = out.withColumn("dagmm_vec", array_to_vector("dagmm_arr"))

    dagmm_gmm = GaussianMixture(featuresCol="dagmm_vec", predictionCol="dagmm_cluster",
                                 probabilityCol="dagmm_probability", k=5, seed=RANDOM_SEED)
    out = dagmm_gmm.fit(out).transform(out)
    out = out.withColumn(
        "dagmm_energy", -F.log(F.array_max(vector_to_array(F.col("dagmm_probability"))) + F.lit(1e-12))
    )
    out = threshold_flag(out, "dagmm_energy", "dagmm_flag", n_std=3.0)

    save_algo_output(out.select("ID", "dagmm_energy", "dagmm_flag"), "dagmm")
    spark.stop()


if __name__ == "__main__":
    main()