
 # Databricks notebook source
from nyctaxi.data_processor import DataProcessor
import yaml

#from pyspark.sql import SparkSession
#spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# Load configuration
with open("project_config.yml", "r") as file:
    config = yaml.safe_load(file)

print("Configuration loaded:")
print(yaml.dump(config, default_flow_style=False))
#config = ProjectConfig.from_yaml(config_path="../../project_config.yml")

# COMMAND ----------
# Load the house prices dataset
#df = spark.read.csv(
    #"/Volumes/mlops_dev/nyctaxis/data/data.csv",
    #header=True,
    #inferSchema=True).toPandas()

#tablepath = "samples.nyctaxi.trips"
#df = spark.sql(f"SELECT * FROM {tablepath}").toPandas()

# COMMAND ----------

data_processor = DataProcessor("samples.nyctaxi.trips", config)
#data_processor = DataProcessor(pandas_df=df, config=config)
data_processor.preprocess_data()
#train_set, test_set = data_processor.split_data()
train_set, test_set = data_processor.split_data()

data_processor.save_to_catalog(train_set=train_set, test_set=test_set, spark=spark)