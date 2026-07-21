"""
task_rl_adaptive_threshold.py - Algorithm 14: RL-tuned Adaptive
Threshold. Only touches raw feature columns from task_prepare_features.py -
no shard or DDP dependency, so it's cheap to retry.
"""
from pyspark.sql import functions as F

from common import FEATURE_COLS, FEATURES_PATH, get_spark_session, rl_tune_adaptive_threshold, save_algo_output


def main():
    spark = get_spark_session("anomaly_rl_adaptive_threshold")
    df = spark.read.format("delta").load(FEATURES_PATH)

    chosen_threshold, stats_row = rl_tune_adaptive_threshold(df, FEATURE_COLS, target_rate=0.02)

    z_conditions = []
    for c in FEATURE_COLS:
        mean_c, std_c = stats_row[f"{c}_mean"], max(stats_row[f"{c}_std"], 1e-10)
        z_conditions.append(F.abs((F.col(c) - F.lit(mean_c)) / F.lit(std_c)) > F.lit(chosen_threshold))
    adaptive_condition = z_conditions[0]
    for cond in z_conditions[1:]:
        adaptive_condition = adaptive_condition | cond

    out = df.withColumn("adaptive_flag", adaptive_condition.cast("int"))
    save_algo_output(out.select("ID", "adaptive_flag"), "rl_adaptive_threshold")
    spark.stop()


if __name__ == "__main__":
    main()