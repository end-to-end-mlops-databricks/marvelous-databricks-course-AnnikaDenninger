# Databricks notebook source
# MAGIC %pip install /Volumes/main/default/file_exchange/denninger/mlops_with_databricks-0.0.1-py3-none-any.whl

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install pyyaml
# MAGIC import yaml

# COMMAND ----------

spark = DatabricksSession.builder.profile("adb-6130442328907134").getOrCreate()

# COMMAND ----------

from scr.nyctaxi.data_processor import DataProcessor

# COMMAND ----------

# Load configuration
with open("project_config.yml", "r") as file:
    config = yaml.safe_load(file)

print("Configuration loaded:")
print(yaml.dump(config, default_flow_style=False))

# COMMAND ----------

data_processor = DataProcessor("dbfs:/databricks-datasets/nyctaxi-with-zipcodes/subsampled", config)


# COMMAND ----------

#df = spark.read.format("delta").load("dbfs:/databricks-datasets/nyctaxi-with-zipcodes/subsampled")

# COMMAND ----------

#df.display()

# COMMAND ----------

data_processor.preprocess_data()

# COMMAND ----------

X_train, X_test, y_train, y_test = data_processor.split_data()
