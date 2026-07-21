"""
task_gmm.py - Algorithm 2: Gaussian Mixture Model.
Native pyspark.ml. Depends only on task_prepare_features.py.
"""
from pyspark.sql import functions as F
from pyspark.ml.clustering import GaussianMixture
from pyspark.ml.functions import array_to_vector

from common import Z_COLS, FEATURES_PATH, RANDOM_SEED, get_spark_session, minority_cluster_flag, save_algo_output


def main():
    spark = get_spark_session("anomaly_gmm")
    df = spark.read.format("delta").load(FEATURES_PATH)
    df = df.withColumn("scaled_features", array_to_vector(F.array(*Z_COLS)))

    gmm = GaussianMixture(featuresCol="scaled_features", predictionCol="gmm_cluster",
                           probabilityCol="gmm_probability", k=6, seed=RANDOM_SEED)
    out = gmm.fit(df).transform(df)
    out = minority_cluster_flag(out, "gmm_cluster", "gmm_flag")

    save_algo_output(out.select("ID", "gmm_cluster", "gmm_flag"), "gmm")
    spark.stop()


if __name__ == "__main__":
    main()