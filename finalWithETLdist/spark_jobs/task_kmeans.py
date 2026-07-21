"""
task_kmeans.py - Algorithm 1: KMeans + Micro-Cluster.
Native pyspark.ml, always was fully distributed. Depends only on
task_prepare_features.py.
"""
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml.clustering import KMeans
from pyspark.ml.functions import array_to_vector

from common import Z_COLS, FEATURES_PATH, RANDOM_SEED, get_spark_session, threshold_flag, save_algo_output


def main():
    spark = get_spark_session("anomaly_kmeans")
    df = spark.read.format("delta").load(FEATURES_PATH)
    df = df.withColumn("scaled_features", array_to_vector(F.array(*Z_COLS)))

    kmeans = KMeans(featuresCol="scaled_features", predictionCol="kmeans_cluster", k=2, seed=RANDOM_SEED)
    model = kmeans.fit(df)
    centers_bc = spark.sparkContext.broadcast(np.array([c.tolist() for c in model.clusterCenters()]))

    @F.pandas_udf(DoubleType())
    def dist_to_nearest_center(*cols):
        X = np.column_stack([c.to_numpy(dtype=np.float64) for c in cols])
        C = centers_bc.value
        d = np.sqrt(((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2))
        return pd.Series(d.min(axis=1))

    out = model.transform(df).withColumn("kmeans_dist", dist_to_nearest_center(*Z_COLS))
    out = threshold_flag(out, "kmeans_dist", "kmeans_flag", n_std=2.0)

    save_algo_output(out.select("ID", "kmeans_dist", "kmeans_flag"), "kmeans")
    spark.stop()


if __name__ == "__main__":
    main()