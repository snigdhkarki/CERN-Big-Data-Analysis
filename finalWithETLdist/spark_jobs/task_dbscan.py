"""
task_dbscan.py - Algorithm 4: DBSCAN, approximated by fitting one local
model per Spark partition in parallel and unioning the core-point sets.
Depends only on task_prepare_features.py.
"""
from pyspark.sql import functions as F

from common import (
    Z_COLS, FEATURES_PATH, NUM_WORKERS,
    get_spark_session, distributed_dbscan_core_points, make_dbscan_udf, save_algo_output,
)


def main():
    spark = get_spark_session("anomaly_dbscan")
    df = spark.read.format("delta").load(FEATURES_PATH)

    core_points, eps = distributed_dbscan_core_points(spark, df, Z_COLS, num_partitions=NUM_WORKERS)
    dbscan_udf = make_dbscan_udf(spark, core_points, eps)

    out = df.withColumn("dbscan_dist", dbscan_udf(*Z_COLS))
    out = out.withColumn("dbscan_flag", (F.col("dbscan_dist") > F.lit(eps)).cast("int"))

    save_algo_output(out.select("ID", "dbscan_dist", "dbscan_flag"), "dbscan")
    spark.stop()


if __name__ == "__main__":
    main()