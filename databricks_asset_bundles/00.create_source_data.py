from pyspark.sql import SparkSession
from pyspark.sql.types import *

# Create SparkSession
spark = SparkSession.builder.getOrCreate()

import pandas as pd
import numpy as np
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from nyctaxi.config import ProjectConfig


# Load configuration
config = ProjectConfig.from_yaml(config_path="project_config.yml")

catalog_name = config.catalog_name
schema_name = config.schema_name

# Load train and test sets
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set_an").toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set_an").toPandas()
combined_set = pd.concat([train_set, test_set], ignore_index=True)
num_rows = len(combined_set)

print(combined_set)
#existing_ids = set(int(id) for id in combined_set['Id'])

# Define function to create synthetic data without random state
def create_synthetic_data(df, num_rows=100):
    synthetic_data = pd.DataFrame()
    
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            mean, std = df[column].mean(), df[column].std()
            synthetic_data[column] = np.random.normal(mean, std, num_rows)
        
        elif pd.api.types.is_categorical_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
            synthetic_data[column] = np.random.choice(df[column].unique(), num_rows, 
                                                      p=df[column].value_counts(normalize=True))
        
        #elif pd.api.types.is_datetime64_any_dtype(df[column]):
            #min_date, max_date = df[column].min(), df[column].max()
            #if min_date < max_date:
                #synthetic_data[column] = pd.to_datetime(
                    #np.random.randint(min_date.value, max_date.value, num_rows)
                #)
            #else:
                #synthetic_data[column] = [min_date] * num_rows
        
        else:
            synthetic_data[column] = np.random.choice(df[column], num_rows)
    
    #new_ids = []
    #i = max(existing_ids) + 1 if existing_ids else 1
    #while len(new_ids) < num_rows:
        #if i not in existing_ids:
            #new_ids.append(str(i))  # Convert numeric ID to string
        #i += 1
    #synthetic_data['Id'] = new_ids

    return synthetic_data

# Create synthetic data
synthetic_df = create_synthetic_data(combined_set)
#print(synthetic_df)

# Define the predefined schema
predefined_schema = StructType([
    StructField("tpep_pickup_datetime", TimestampType(), True),
    StructField("tpep_dropoff_datetime", TimestampType(), True),
    StructField("trip_distance", DoubleType(), True),
    StructField("fare_amount", DoubleType(), True),
    StructField("pickup_zip", IntegerType(), True),
    StructField("dropoff_zip", IntegerType(), True),
    StructField("update_timestamp_utc", TimestampType(), True),
])

# Create the table if it does not exist
if not spark.catalog.tableExists("sandbox.sb_adan.source_data_an"):
    # Create an empty DataFrame with the same columns and data types as synthetic_df
    empty_df = spark.createDataFrame([], predefined_schema)
    empty_df.write.saveAsTable("sandbox.sb_adan.source_data_an")

print("Load the existing schema")
existing_schema = spark.table(f"{catalog_name}.{schema_name}.source_data_an").schema

print("Write dataset")
synthetic_spark_df = spark.createDataFrame(synthetic_df, schema=existing_schema)

print("Add timestamp")
train_set_with_timestamp = synthetic_spark_df.withColumn(
    "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
)

print("Append synthetic data as new data to source_data table")
train_set_with_timestamp.write.mode("append").saveAsTable(
    "sandbox.sb_adan.source_data_an"
)

print("done")
