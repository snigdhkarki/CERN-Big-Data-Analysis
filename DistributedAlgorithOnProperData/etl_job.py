# from pyspark.sql import SparkSession

# # Initialize Spark with the Oracle JDBC driver
# spark = SparkSession.builder \
#     .appName("OracleToHDFS_ETL") \
#     .config("spark.jars", "ojdbc8.jar") \
#     .getOrCreate()

# # 1. Extract: Read data from Oracle
# oracle_df = spark.read \
#     .format("jdbc") \
#     .option("url", "jdbc:oracle:thin:@localhost:1521/XE") \
#     .option("dbtable", "SYSTEM.NETWORK_TRAFFIC") \
#     .option("user", "system") \
#     .option("password", "demo_password") \
#     .option("driver", "oracle.jdbc.driver.OracleDriver") \
#     .load()

# print("Data extracted from Oracle:")
# oracle_df.show()

# # 2. Load: Write to HDFS as Parquet
# hdfs_path = "hdfs://localhost:9000/user/demo/network_data.parquet"
# oracle_df = oracle_df.repartition(4) 
# oracle_df.write.mode("overwrite").parquet(hdfs_path)

# print(f"Data successfully written to {hdfs_path}")
# spark.stop()

from pyspark.sql import SparkSession
from pyspark.sql.functions import row_number, lit
from pyspark.sql.window import Window

spark = SparkSession.builder \
    .appName("OracleToHDFS_ETL") \
    .config("spark.jars", "ojdbc8.jar") \
    .getOrCreate()

# 1. Extract: read all columns from Oracle
oracle_df = spark.read \
    .format("jdbc") \
    .option("url", "jdbc:oracle:thin:@localhost:1521/XE") \
    .option("dbtable", "SYSTEM.NETWORK_TRAFFIC") \
    .option("user", "system") \
    .option("password", "demo_password") \
    .option("driver", "oracle.jdbc.driver.OracleDriver") \
    .load()

# Add an ID column (if the table doesn't have one already)
# Since we used IDENTITY, it's there, but we can renumber for safety
window = Window.orderBy(lit(1))
oracle_df = oracle_df.withColumn("ID", row_number().over(window))

# Select only the needed columns: ID and the 5 features
# (Column names from Oracle are uppercase by default)
oracle_df = oracle_df.select("ID", "PIXEL_TEMP_C", "LAR_PRESSURE_BAR",
                             "LV_VOLTAGE_V", "HV_CURRENT_UA", "COOLING_FLOW_LPM")

print("Data extracted from Oracle (first 5 rows):")
oracle_df.show(5)

# 2. Write to HDFS as Parquet
hdfs_path = "hdfs://localhost:9000/user/demo/network_data.parquet"
oracle_df = oracle_df.repartition(4)
oracle_df.write.mode("overwrite").parquet(hdfs_path)

print(f"Data successfully written to {hdfs_path}")
spark.stop()