# Databricks notebook source
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %pip install /Volumes/main/default/file_exchange/denninger/mlops_with_databricks-0.0.1-py3-none-any.whl

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install pyyaml
# MAGIC import yaml

# COMMAND ----------

from scr.nyctaxi.data_processor import DataProcessor

# COMMAND ----------

# Load configuration
with open("project_config.yml", "r") as file:
    config = yaml.safe_load(file)

print("Configuration loaded:")
print(yaml.dump(config, default_flow_style=False))

# COMMAND ----------

data_processor = DataProcessor(spark, "dbfs:/databricks-datasets/nyctaxi-with-zipcodes/subsampled", config)


# COMMAND ----------

data_processor.preprocess_data()

# COMMAND ----------

train_data, test_data = data_processor.split_data()

print("Training set shape:", train_data.count(), "rows")
print("Test set shape:", test_data.count(), "rows")

# COMMAND ----------


