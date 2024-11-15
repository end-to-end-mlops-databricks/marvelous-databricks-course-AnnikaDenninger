"""
This script handles data ingestion and feature table updates for a house price prediction system.

Key functionality:
- Loads the source dataset and identifies new records for processing
- Splits new records into train and test sets based on timestamp
- Updates existing train and test tables with new data
- Inserts the latest feature values into the feature table for serving
- Triggers and monitors pipeline updates for online feature refresh
- Sets task values to coordinate pipeline orchestration

Workflow:
1. Load source dataset and retrieve recent records with updated timestamps.
2. Split new records into train and test sets (80-20 split).
3. Append new train and test records to existing train and test tables.
4. Insert the latest feature data into the feature table for online serving.
5. Trigger a pipeline update and monitor its status until completion.
6. Set a task value indicating whether new data was processed.
"""

import yaml
import argparse
from pyspark.sql import SparkSession
import yaml
from databricks.sdk import WorkspaceClient
import time
from nyctaxi.config import ProjectConfig

workspace = WorkspaceClient()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)

args = parser.parse_args()
root_path = args.root_path
config_path = (f"{root_path}/project_config.yml")
config = ProjectConfig.from_yaml(config_path=config_path)
pipeline_id = config.pipeline_id

spark = SparkSession.builder.getOrCreate()

catalog_name = config.catalog_name
schema_name = config.schema_name

# Load source_data table
source_data = spark.table(f"{catalog_name}.{schema_name}.source_data_an")

# Get max update timestamps from existing data
max_train_timestamp = spark.table(f"{catalog_name}.{schema_name}.train_set_an") \
    .select(spark_max("update_timestamp_utc").alias("max_update_timestamp")) \
    .collect()[0]["max_update_timestamp"]

max_test_timestamp = spark.table(f"{catalog_name}.{schema_name}.test_set_an") \
    .select(spark_max("update_timestamp_utc").alias("max_update_timestamp")) \
    .collect()[0]["max_update_timestamp"]

latest_timestamp = max(max_train_timestamp, max_test_timestamp)

# Filter source_data for rows with update_timestamp_utc greater than the latest_timestamp
new_data = source_data.filter(col("update_timestamp_utc") > latest_timestamp)

# Split the new data into train and test sets
new_data_train, new_data_test = new_data.randomSplit([0.8, 0.2], seed=42)

# Update train_set and test_set tables
new_data_train.write.mode("append").saveAsTable(f"{catalog_name}.{schema_name}.train_set_an_new")
new_data_test.write.mode("append").saveAsTable(f"{catalog_name}.{schema_name}.test_set_an_new")

# Verify affected rows count for train and test
affected_rows_train = new_data_train.count()
affected_rows_test = new_data_test.count()

#write into feature table; update online table
if affected_rows_train > 0 or affected_rows_test > 0 :
    spark.sql(f"""
        WITH max_timestamp AS (
            SELECT MAX(update_timestamp_utc) AS max_update_timestamp
            FROM {catalog_name}.{schema_name}.train_set_an_new
        )
        INSERT INTO {catalog_name}.{schema_name}.features_an
        SELECT pickup_zip, trip_distance
        FROM {catalog_name}.{schema_name}.train_set_an_new
        WHERE update_timestamp_utc == (SELECT max_update_timestamp FROM max_timestamp)
""")
    spark.sql(f"""
        WITH max_timestamp AS (
            SELECT MAX(update_timestamp_utc) AS max_update_timestamp
            FROM {catalog_name}.{schema_name}.test_set_an_new
        )
        INSERT INTO {catalog_name}.{schema_name}.features_an
        SELECT pickup_zip, trip_distance
        FROM {catalog_name}.{schema_name}.test_set_an_new
        WHERE update_timestamp_utc == (SELECT max_update_timestamp FROM max_timestamp)
""")
    refreshed = 1
    update_response = workspace.pipelines.start_update(
        pipeline_id=pipeline_id, full_refresh=False)
    while True:
        update_info = workspace.pipelines.get_update(pipeline_id=pipeline_id, 
                                update_id=update_response.update_id)
        state = update_info.update.state.value
        if state == 'COMPLETED':
            break
        elif state in ['FAILED', 'CANCELED']:
            raise SystemError("Online table failed to update.")
        elif state == 'WAITING_FOR_RESOURCES':
            print("Pipeline is waiting for resources.")
        else:
            print(f"Pipeline is in {state} state.")
        time.sleep(30)
else:
    refreshed = 0

dbutils.jobs.taskValues.set(key="refreshed", value=refreshed)