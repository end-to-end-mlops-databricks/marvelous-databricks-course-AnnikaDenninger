# Databricks notebook source
# MAGIC %pip install /Volumes/main/default/file_exchange/denninger/nyc_taxi-0.0.1-py3-none-any.whl

# COMMAND ----------
# MAGIC %restart_python

# COMMAND ----------

import time

import requests
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    TrafficConfig,
    Route,
)

from nyctaxi.config import ProjectConfig
from pyspark.sql import SparkSession

workspace = WorkspaceClient()
spark = SparkSession.builder.getOrCreate()

config = ProjectConfig.from_yaml(config_path="project_config.yml")

catalog_name = config.catalog_name
schema_name = config.schema_name

train_set = spark.table(f"{catalog_name}.{schema_name}.train_set_an").toPandas()

workspace.serving_endpoints.create(
    name="nyctaxi-model-serving",
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=f"{catalog_name}.{schema_name}.nyctaxi_model_basic",
                scale_to_zero_enabled=True,
                workload_size="Small",
                entity_version=3,
            )
        ],
    # Optional if only 1 entity is served
    traffic_config=TrafficConfig(
        routes=[
            Route(served_model_name="nyctaxi_model_basic-3",
                  traffic_percentage=100)
        ]
        ),
    ),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Call the endpoint

# COMMAND ----------

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create sample request body

# COMMAND ----------

required_columns = [
    "trip_distance"
]

sampled_records = train_set[required_columns].sample(n=1000, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

# COMMAND ----------
dataframe_records

# COMMAND ----------

"""
Each body should be list of json with columns

Example

[{'trip_distance': 2.0
  }]
"""

# COMMAND ----------
start_time = time.time()

model_serving_endpoint = (
    f"https://{host}/serving-endpoints/nyctaxi-model-serving/invocations"
)
response = requests.post(
    f"{model_serving_endpoint}",
    headers={"Authorization": f"Bearer {token}"},
    json={"dataframe_records": dataframe_records[0]},
)

end_time = time.time()
execution_time = end_time - start_time

print("Response status:", response.status_code)
print("Reponse text:", response.text)
print("Execution time:", execution_time, "seconds")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Test

# COMMAND ----------

# Initialize variables
model_serving_endpoint = (
    f"https://{host}/serving-endpoints/nyctaxi-model-serving/invocations"
)

headers = {"Authorization": f"Bearer {token}"}
num_requests = 10


# Function to make a request and record latency
def send_request():
    random_record = random.choice(dataframe_records)
    start_time = time.time()
    response = requests.post(
        model_serving_endpoint,
        headers=headers,
        json={"dataframe_records": random_record},
    )
    end_time = time.time()
    latency = end_time - start_time
    return response.status_code, latency


total_start_time = time.time()
latencies = []

# Send requests concurrently
with ThreadPoolExecutor(max_workers=100) as executor:
    futures = [executor.submit(send_request) for _ in range(num_requests)]

    for future in as_completed(futures):
        status_code, latency = future.result()
        latencies.append(latency)

total_end_time = time.time()
total_execution_time = total_end_time - total_start_time

# Calculate the average latency
average_latency = sum(latencies) / len(latencies)

print("\nTotal execution time:", total_execution_time, "seconds")
print("Average latency per request:", average_latency, "seconds")