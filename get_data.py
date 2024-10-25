
# Databricks notebook source

# COMMAND ----------
 
from databricks.connect import DatabricksSession

spark = DatabricksSession.builder.profile("adb-6130442328907134").getOrCreate()
#path = "sandbox.sb_adan.tetuan_city_power_consumption"
path = "samples.nyctaxi.trips"

df = spark.read.table(path)

df.show(5)
 
# COMMAND ----------
