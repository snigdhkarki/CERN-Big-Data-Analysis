from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
import pyspark.sql.functions as F

spark = SparkSession.builder.appName("HDFS_AnomalyDetection").getOrCreate()

# 1. Read the Parquet data directly from Hadoop
hdfs_path = "hdfs://192.168.18.4:9000/user/demo/network_data.parquet"
df = spark.read.parquet(hdfs_path)

# 2. Feature Engineering
assembler = VectorAssembler(
    inputCols=["PACKET_SIZE", "CONNECTION_TIME", "FAILED_LOGINS"], 
    outputCol="features"
)
feature_df = assembler.transform(df)

# Scale the features so packet size doesn't dominate failed logins
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
scaled_df = scaler.fit(feature_df).transform(feature_df)

# 3. Train K-Means Model (k=2)
kmeans = KMeans(featuresCol="scaledFeatures", k=2, seed=42)
model = kmeans.fit(scaled_df)
predictions = model.transform(scaled_df)

# 4. Identify the Anomaly (Micro-Cluster approach)
# Count how many data points are assigned to each cluster
cluster_counts = predictions.groupBy("prediction").count()

# Join the counts back and flag any row belonging to a cluster with only 1 point
final_df = predictions.join(cluster_counts, "prediction") \
    .withColumn("Is_Anomaly", F.when(F.col("count") == 1, True).otherwise(False)) \
    .orderBy("ID")

print("Anomaly Detection Results (Using K-Means Micro-Clusters):")
# We select 'prediction' (cluster ID) so you can show the professor how the algorithm separated them
final_df.select("ID", "PACKET_SIZE", "FAILED_LOGINS", "prediction", "Is_Anomaly").show()

spark.stop()


