"""
task_isolation_forest.py - Algorithm 5: Isolation Forest, approximated
by fitting one small forest per Spark partition in parallel and
ensemble-averaging their scores. Depends only on task_prepare_features.py.
"""
from common import (
    Z_COLS, FEATURES_PATH, NUM_WORKERS,
    get_spark_session, distributed_isolation_forest, make_iso_ensemble_udf, threshold_flag, save_algo_output,
)


def main():
    spark = get_spark_session("anomaly_isolation_forest")
    df = spark.read.format("delta").load(FEATURES_PATH)

    forests = distributed_isolation_forest(spark, df, Z_COLS, num_partitions=NUM_WORKERS)
    iso_udf = make_iso_ensemble_udf(spark, forests)

    out = df.withColumn("iso_score", iso_udf(*Z_COLS))
    out = threshold_flag(out, "iso_score", "iso_flag", n_std=2.6)

    save_algo_output(out.select("ID", "iso_score", "iso_flag"), "isolation_forest")
    spark.stop()


if __name__ == "__main__":
    main()