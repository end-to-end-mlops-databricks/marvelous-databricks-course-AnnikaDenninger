from pyspark.sql.functions import col
from databricks.connect import DatabricksSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, ArrayType
from databricks.sdk import WorkspaceClient

from nyctaxi.config import ProjectConfig

spark = DatabricksSession.builder.getOrCreate()
workspace = WorkspaceClient()

# Load configuration
config = ProjectConfig.from_yaml(config_path="project_config.yml")
catalog_name = config.catalog_name
schema_name = config.schema_name

inf_table = spark.sql(f"SELECT * FROM {catalog_name}.{schema_name}.`model_serving_fe_payload`")

print("create schemas")
request_schema = StructType([
    StructField("dataframe_records", ArrayType(StructType([
        StructField("trip_distance", DoubleType(), True),
        StructField("pickup_zip", StringType(), True)
    ])), True)  
])

response_schema = StructType([
    StructField("predictions", ArrayType(DoubleType()), True),
    StructField("databricks_output", StructType([
        StructField("trace", StringType(), True),
        StructField("databricks_request_id", StringType(), True)
    ]), True)
])

print("draw request and response")
inf_table_parsed = inf_table.withColumn("parsed_request", 
                                        F.from_json(F.col("request"),
                                                    request_schema))


inf_table_parsed = inf_table_parsed.withColumn("parsed_response",
                                               F.from_json(F.col("response"),
                                                           response_schema))


df_exploded = inf_table_parsed.withColumn("record",
                                          F.explode(F.col("parsed_request.dataframe_records")))


print("create table with model inputs and predictions")
df_final = df_exploded.select(
    F.from_unixtime(F.col("timestamp_ms") / 1000).cast("timestamp").alias("timestamp"),
    "timestamp_ms",
    "databricks_request_id",
    "execution_time_ms",
    F.col("record.pickup_zip").alias("pickup_zip"),
    F.col("record.trip_distance").alias("trip_distance"),
    F.col("parsed_response.predictions")[0].alias("prediction"),
    F.lit("nyctaxi_model_fe").alias("model_name")
)

test_set = spark.table(f"{catalog_name}.{schema_name}.train_set_an")
inference_set_normal = spark.table(f"{catalog_name}.{schema_name}.inference_set_normal_an")
inference_set_skewed = spark.table(f"{catalog_name}.{schema_name}.inference_set_skewed_an")

inference_set = inference_set_normal.union(inference_set_skewed)

print("Add fare_amount to table")
df_final_with_status = df_final \
    .join(test_set.select("pickup_zip", "fare_amount"), on="pickup_zip", how="left") \
    .withColumnRenamed("fare_amount", "fare_amount_test") \
    .join(inference_set.select("pickup_zip", "fare_amount"), on="pickup_zip", how="left") \
    .withColumnRenamed("fare_amount", "fare_amount_inference") \
    .select(
        "*",  
        F.coalesce(F.col("fare_amount_test"), F.col("fare_amount_inference")).alias("fare_amount")
    ) \
    .drop("fare_amount_test", "fare_amount_inference") \
    .withColumn("fare_amount", F.col("fare_amount").cast("double")) \
    .withColumn("prediction", F.col("prediction").cast("double")) \
    .dropna(subset=["fare_amount", "prediction"])


df_final_with_status.write.format("delta").mode("append")\
    .saveAsTable(f"{catalog_name}.{schema_name}.model_monitoring_an")

workspace.quality_monitors.run_refresh(
    table_name=f"{catalog_name}.{schema_name}.model_monitoring_an"
)