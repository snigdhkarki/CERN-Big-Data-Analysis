# from pyspark.sql import SparkSession
# from pyspark.sql.functions import row_number, lit
# from pyspark.sql.window import Window
# from delta import configure_spark_with_delta_pip
#
# ... (all commented-out, leave as-is) ...

from pyspark.sql import SparkSession
import pandas as pd
from delta import configure_spark_with_delta_pip

spark = SparkSession.builder.appName("OracleToHDFS_ETL").getOrCreate()

# 1. Extract
oracle_df = spark.read \
    .format("jdbc") \
    .option("url", "jdbc:oracle:thin:@oracle:1521/XE") \
    .option("dbtable", "SYSTEM.NETWORK_TRAFFIC") \
    .option("user", "system") \
    .option("password", "demo_password") \
    .option("driver", "oracle.jdbc.driver.OracleDriver") \
    .load()

oracle_df = oracle_df.select("ID", "PIXEL_TEMP_C", "LAR_PRESSURE_BAR",
                              "LV_VOLTAGE_V", "HV_CURRENT_UA", "COOLING_FLOW_LPM") \
                      .orderBy("ID")

print("Data extracted from Oracle (first 5 rows):")
oracle_df.show(5)

CHANNELS = ["PIXEL_TEMP_C", "LAR_PRESSURE_BAR", "LV_VOLTAGE_V",
            "HV_CURRENT_UA", "COOLING_FLOW_LPM"]

from pyspark.sql.types import StructType, StructField, LongType, DoubleType

pdf = oracle_df.toPandas()
pdf[CHANNELS] = pdf[CHANNELS].astype(float)

n_missing = pdf[CHANNELS].isna().sum().sum()
if n_missing:
    print(f"Found {n_missing} missing values across {CHANNELS} -- interpolating")

pdf[CHANNELS] = (
    pdf[CHANNELS]
    .interpolate(method="linear", limit_direction="both")
    .ffill()
    .bfill()
)

remaining = pdf[CHANNELS].isna().sum().sum()
if remaining:
    raise RuntimeError(f"{remaining} values could not be reconstructed")

clean_schema = StructType(
    [StructField("ID", LongType(), True)]
    + [StructField(c, DoubleType(), True) for c in CHANNELS]
)
pdf["ID"] = pdf["ID"].astype("int64")
oracle_df = spark.createDataFrame(pdf, schema=clean_schema)

hdfs_path = "hdfs://namenode:9000/user/demo/network_data"
oracle_df = oracle_df.repartition(4)
oracle_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(hdfs_path)

print(f"Data successfully written to {hdfs_path}")
spark.stop()