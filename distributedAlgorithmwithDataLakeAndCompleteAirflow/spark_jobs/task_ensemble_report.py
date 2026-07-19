"""
task_ensemble_report.py

Final task in the DAG. Depends on all 14 algorithm tasks. Joins every
algorithm's per-ID flag back onto the feature table, computes the
ensemble vote, prints the same reporting the monolithic script used to
print, and persists the final labeled table to HDFS. Cheap to retry -
it only does joins/aggregations over already-computed flags, never
retrains anything.
"""
import math

from pyspark.sql import functions as F

from common import (
    FEATURE_COLS, FEATURES_PATH, FINAL_OUTPUT_PATH, ALGO_FLAG_COLS,
    get_spark_session, load_algo_output, active_exploration_queue,
)


def main():
    spark = get_spark_session("anomaly_ensemble_report")

    working_df = spark.read.format("delta").load(FEATURES_PATH).select("ID", *FEATURE_COLS)
    for algo_name in ALGO_FLAG_COLS:
        algo_df = load_algo_output(spark, algo_name)
        # some algo outputs carry extra diagnostic columns (scores,
        # cluster ids) alongside their flag - only the flag matters here
        flag_col = ALGO_FLAG_COLS[algo_name]
        working_df = working_df.join(algo_df.select("ID", flag_col), on="ID", how="left")

    # rows missing from an algo's output (e.g. LSTM/OmniAnomaly boundary
    # rows that fall outside any full sliding window) default to "not
    # flagged by that algorithm" rather than nulling out the vote.
    working_df = working_df.fillna({c: 0 for c in ALGO_FLAG_COLS.values()})

    flag_cols = list(ALGO_FLAG_COLS.values())
    n_algorithms = len(flag_cols)
    min_votes = math.ceil(n_algorithms / 4)

    vote_expr = F.col(flag_cols[0])
    for c in flag_cols[1:]:
        vote_expr = vote_expr + F.col(c)

    final_df = working_df.withColumn("vote", vote_expr).withColumn(
        "ensemble", (F.col("vote") >= F.lit(min_votes)).cast("int")
    ).cache()

    explore_df = active_exploration_queue(final_df, n_algorithms)

    print(f"\n[Driver] {n_algorithms} algorithms in the ensemble; flagging anomaly at vote >= {min_votes}.")

    print("\n--- Top 10 Most Anomalous Rows (Highest Votes) ---")
    final_df.select("ID", *FEATURE_COLS, "vote", "ensemble").orderBy(F.desc("vote")).limit(10).show(truncate=False)

    print("\n--- Algorithm Performance Summary ---")
    summary_exprs = [F.sum(c).alias(c) for c in flag_cols] + [F.sum("ensemble").alias("Ensemble_Final")]
    print(final_df.select(*summary_exprs).first().asDict())

    print("\n--- Cluster Sizes (Ensemble) ---")
    final_df.groupBy("ensemble").count().show()

    print(f"\n--- Active Exploration Queue (top {explore_df.count()} rows for analyst review) ---")
    explore_df.select("ID", *FEATURE_COLS, "vote", "boundary_distance").orderBy("boundary_distance").show(truncate=False)

    (final_df.select("ID", *FEATURE_COLS, *flag_cols, "vote", "ensemble")
             .write.format("delta").mode("overwrite").save(FINAL_OUTPUT_PATH))
    print(f"[Driver] Wrote final ensemble table -> {FINAL_OUTPUT_PATH}")

    spark.stop()


if __name__ == "__main__":
    main()