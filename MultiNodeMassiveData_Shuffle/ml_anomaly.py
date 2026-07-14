"""
Distributed Anomaly Ensemble - PySpark ML / SQL Edition
==========================================================

Same four algorithms as the no-shuffle version, same ensemble rule
(vote >= 2 flagged components -> anomaly), but rebuilt on top of
pyspark.ml Estimators/Transformers and pyspark.sql instead of hand-rolled
mapPartitions / aggregate / broadcast plumbing.

This version assumes shuffle read/write between workers is healthy (true
now that we're on a bridge network instead of host networking), so we're
free to use GroupBy, OrderBy, and MLlib's built-in fit()/transform(),
which internally use treeAggregate and shuffle-based joins.

    1. KMeans (k=2)             -> distance to nearest centroid,
                                    threshold = mean + 3*std
    2. GaussianMixture (k=3)    -> flag rows in the minority (least
                                    populous) mixture component
    3. PCA(k=2) + KMeans (k=2)  -> flag rows in the minority PCA-space
                                    cluster
    4. Adaptive Threshold       -> per-feature z-score, flag if any
                                    |z| > 3

Note on Algorithm 3: the old version fit local K-Means (k=4) per
partition and merged the resulting centers on the driver via a second
K-Means pass, as an approximation of a global fit without a real
shuffle. Here we just fit KMeans(k=2) directly on the full PCA-projected
DataFrame - a real global fit, and simpler, now that a shuffle is cheap.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml.clustering import KMeans, GaussianMixture
import numpy as np

FEATURE_COLS = ["PACKET_SIZE", "CONNECTION_TIME", "FAILED_LOGINS"]
N_FEATURES = len(FEATURE_COLS)
ADAPTIVE_Z = 3.0


def main():
    spark = SparkSession.builder.appName("Distributed_Anomaly_Ensemble_PySparkML").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # ------------------------------------------------------------------
    # 0. Load + assemble + scale
    # ------------------------------------------------------------------
    hdfs_path = "hdfs://192.168.18.4:9000/user/demo/network_data.parquet"
    df = spark.read.parquet(hdfs_path).select("ID", *FEATURE_COLS)

    total_rows = df.count()
    print(f"[Driver] Loaded {total_rows} rows from HDFS.")

    assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="raw_features")
    assembled = assembler.transform(df)

    scaler = StandardScaler(
        inputCol="raw_features", outputCol="scaled_features",
        withMean=True, withStd=True,
    )
    scaled_df = scaler.fit(assembled).transform(assembled).cache()

    # ------------------------------------------------------------------
    # ALGORITHM 1: KMeans + Micro-Cluster
    # ------------------------------------------------------------------
    print("\n[MLlib] Algorithm 1: KMeans + Micro-Cluster...")
    kmeans = KMeans(featuresCol="scaled_features", predictionCol="kmeans_cluster", k=2, seed=42)
    kmeans_model = kmeans.fit(scaled_df)
    centers = np.array([c.tolist() for c in kmeans_model.clusterCenters()])
    print(f"[Driver] KMeans centers:\n{centers}")

    centers_bc = spark.sparkContext.broadcast(centers)

    @F.udf(returnType=DoubleType())
    def dist_to_nearest_center(vec):
        v = np.array(vec)
        return float(np.min(np.linalg.norm(centers_bc.value - v, axis=1)))

    kmeans_df = kmeans_model.transform(scaled_df).withColumn(
        "kmeans_dist", dist_to_nearest_center(F.col("scaled_features"))
    )

    dist_stats = kmeans_df.select(
        F.mean("kmeans_dist").alias("mean"), F.stddev_pop("kmeans_dist").alias("std")
    ).first()
    kmeans_threshold = dist_stats["mean"] + 3 * dist_stats["std"]
    print(f"[Driver] KMeans distance threshold: {kmeans_threshold:.4f}")

    kmeans_df = kmeans_df.withColumn(
        "kmeans_flag", (F.col("kmeans_dist") > F.lit(kmeans_threshold)).cast("int")
    )

    # ------------------------------------------------------------------
    # ALGORITHM 2: Gaussian Mixture Model, flag minority component
    # ------------------------------------------------------------------
    print("\n[MLlib] Algorithm 2: Gaussian Mixture Model...")
    gmm = GaussianMixture(featuresCol="scaled_features", predictionCol="gmm_cluster", k=3, seed=42)
    gmm_df = gmm.fit(scaled_df).transform(kmeans_df)

    gmm_counts_map = {r["gmm_cluster"]: r["count"] for r in gmm_df.groupBy("gmm_cluster").count().collect()}
    gmm_minority = min(gmm_counts_map, key=gmm_counts_map.get)
    print(f"[Driver] GMM cluster counts: {gmm_counts_map}, minority = {gmm_minority}")

    gmm_df = gmm_df.withColumn(
        "gmm_flag", (F.col("gmm_cluster") == F.lit(gmm_minority)).cast("int")
    )

    # ------------------------------------------------------------------
    # ALGORITHM 3: PCA + KMeans Micro-Cluster
    # ------------------------------------------------------------------
    print("\n[MLlib] Algorithm 3: PCA + KMeans Micro-Cluster...")
    pca = PCA(k=2, inputCol="scaled_features", outputCol="pca_features")
    pca_df = pca.fit(scaled_df).transform(gmm_df)

    pca_kmeans = KMeans(featuresCol="pca_features", predictionCol="pca_cluster", k=2, seed=42)
    pca_df = pca_kmeans.fit(pca_df).transform(pca_df)

    pca_counts_map = {r["pca_cluster"]: r["count"] for r in pca_df.groupBy("pca_cluster").count().collect()}
    pca_minority = min(pca_counts_map, key=pca_counts_map.get)
    print(f"[Driver] PCA cluster counts: {pca_counts_map}, minority = {pca_minority}")

    pca_df = pca_df.withColumn(
        "pca_flag", (F.col("pca_cluster") == F.lit(pca_minority)).cast("int")
    )

    # ------------------------------------------------------------------
    # ALGORITHM 4: Adaptive Threshold (per-feature z-score, mean + 3*std)
    # ------------------------------------------------------------------
    print("\n[MLlib] Algorithm 4: Adaptive Threshold...")
    stat_exprs = []
    for c in FEATURE_COLS:
        stat_exprs.append(F.mean(c).alias(f"{c}_mean"))
        stat_exprs.append(F.stddev_pop(c).alias(f"{c}_std"))
    stats_row = df.select(*stat_exprs).first()

    z_conditions = []
    for c in FEATURE_COLS:
        mean_c = stats_row[f"{c}_mean"]
        std_c = max(stats_row[f"{c}_std"], 1e-10)
        print(f"[Driver] {c}: mean={mean_c:.4f} std={std_c:.4f}")
        z_conditions.append(F.abs((F.col(c) - F.lit(mean_c)) / F.lit(std_c)) > ADAPTIVE_Z)

    adaptive_condition = z_conditions[0]
    for cond in z_conditions[1:]:
        adaptive_condition = adaptive_condition | cond

    final_df = pca_df.withColumn("adaptive_flag", adaptive_condition.cast("int"))

    # ------------------------------------------------------------------
    # ENSEMBLE VOTE
    # ------------------------------------------------------------------
    final_df = final_df.withColumn(
        "vote",
        F.col("kmeans_flag") + F.col("gmm_flag") + F.col("pca_flag") + F.col("adaptive_flag"),
    ).withColumn(
        "ensemble", (F.col("vote") >= 2).cast("int")
    ).cache()

    # ------------------------------------------------------------------
    # REPORTING
    # ------------------------------------------------------------------
    print("\n--- Top 10 Most Anomalous Rows (Highest Votes) ---")
    final_df.select("ID", *FEATURE_COLS, "vote", "ensemble") \
        .orderBy(F.desc("vote")) \
        .limit(10) \
        .show(truncate=False)

    print("\n--- Algorithm Performance Summary ---")
    summary = final_df.select(
        F.sum("kmeans_flag").alias("KMeans_Adaptive"),
        F.sum("gmm_flag").alias("GMM_Minority"),
        F.sum("pca_flag").alias("PCA_Embedding"),
        F.sum("adaptive_flag").alias("Adaptive_Threshold"),
        F.sum("ensemble").alias("Ensemble_Final"),
    ).first()
    print(summary.asDict())

    print("\n--- Cluster Sizes (Ensemble) ---")
    final_df.groupBy("ensemble").count().show()

    spark.stop()


if __name__ == "__main__":
    main()