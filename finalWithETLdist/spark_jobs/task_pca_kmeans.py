"""
task_pca_kmeans.py - Algorithm 3: PCA + KMeans Micro-Cluster.
Native pyspark.ml. Depends only on task_prepare_features.py.
"""
from pyspark.sql import functions as F
from pyspark.ml.feature import PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.functions import array_to_vector

from common import Z_COLS, FEATURES_PATH, RANDOM_SEED, get_spark_session, minority_cluster_flag, save_algo_output


def main():
    spark = get_spark_session("anomaly_pca_kmeans")
    df = spark.read.format("delta").load(FEATURES_PATH)
    df = df.withColumn("scaled_features", array_to_vector(F.array(*Z_COLS)))

    pca = PCA(k=2, inputCol="scaled_features", outputCol="pca_features")
    out = pca.fit(df).transform(df)

    pca_kmeans = KMeans(featuresCol="pca_features", predictionCol="pca_cluster", k=4, seed=RANDOM_SEED)
    out = pca_kmeans.fit(out).transform(out)
    out = minority_cluster_flag(out, "pca_cluster", "pca_flag")

    save_algo_output(out.select("ID", "pca_cluster", "pca_flag"), "pca_kmeans")
    spark.stop()


if __name__ == "__main__":
    main()