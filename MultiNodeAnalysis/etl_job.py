from pyspark.sql import SparkSession

# Initialize Spark with the Oracle JDBC driver
spark = SparkSession.builder \
    .appName("OracleToHDFS_ETL") \
    .config("spark.jars", "ojdbc8.jar") \
    .getOrCreate()

# 1. Extract: Read data from Oracle
oracle_df = spark.read \
    .format("jdbc") \
    .option("url", "jdbc:oracle:thin:@localhost:1521/XE") \
    .option("dbtable", "SYSTEM.NETWORK_TRAFFIC") \
    .option("user", "system") \
    .option("password", "demo_password") \
    .option("driver", "oracle.jdbc.driver.OracleDriver") \
    .load()

print("Data extracted from Oracle:")
oracle_df.show()

# 2. Load: Write to HDFS as Parquet
hdfs_path = "hdfs://localhost:9000/user/demo/network_data.parquet"
oracle_df.write.mode("overwrite").parquet(hdfs_path)

print(f"Data successfully written to {hdfs_path}")
spark.stop()