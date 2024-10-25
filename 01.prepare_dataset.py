
 # Databricks notebook source
from nyctaxi.data_processor import DataProcessor
import yaml

# COMMAND ----------

# Load configuration
with open("project_config.yml", "r") as file:
    config = yaml.safe_load(file)

print("Configuration loaded:")
print(yaml.dump(config, default_flow_style=False))

# COMMAND ----------

data_processor = DataProcessor("samples.nyctaxi.trips", config)
#data_processor = DataProcessor(pandas_df=df, config=config)
data_processor.preprocess_data()
#train_set, test_set = data_processor.split_data()
train_set, test_set = data_processor.split_data()

data_processor.save_to_catalog(train_set=train_set, test_set=test_set, spark=spark)