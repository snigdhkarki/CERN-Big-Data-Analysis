"""
task_prepare_features.py

First task in the DAG (after the CSV/Oracle/ETL steps already existed).
Reads the raw Delta table etl_job.py wrote, assembles + standard-scales
the 5 features, and writes ID + raw features + z-scored features to
HDFS as Delta. Every other algorithm task reads from here instead of
repeating this fit - so a retry of, say, the GAN task never re-derives
scaling statistics that could subtly drift from what the other 13
algorithms used.
"""
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.functions import vector_to_array

from common import FEATURE_COLS, Z_COLS, SOURCE_HDFS_PATH, FEATURES_PATH, get_spark_session


def main():
    spark = get_spark_session("anomaly_prepare_features")

    df = spark.read.format("delta").load(SOURCE_HDFS_PATH).select("ID", *FEATURE_COLS)
    total_rows = df.count()
    print(f"[Driver] Loaded {total_rows} rows from {SOURCE_HDFS_PATH}.")

    assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="raw_features")
    assembled = assembler.transform(df)
    scaler = StandardScaler(inputCol="raw_features", outputCol="scaled_features", withMean=True, withStd=True)
    scaled = scaler.fit(assembled).transform(assembled)

    scaled = scaled.withColumn("_z", vector_to_array("scaled_features"))
    for i, c in enumerate(FEATURE_COLS):
        scaled = scaled.withColumn(f"z_{c}", F.col("_z")[i])

    out = scaled.select("ID", *FEATURE_COLS, *Z_COLS)
    out.write.format("delta").mode("overwrite").save(FEATURES_PATH)
    print(f"[Driver] Wrote {total_rows} rows of engineered features -> {FEATURES_PATH}")

    spark.stop()


if __name__ == "__main__":
    main()