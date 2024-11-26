# Databricks notebook source
# MAGIC %md
# MAGIC # Generate synthetic datasets for inference

# COMMAND ----------

# MAGIC %pip install /Volumes/main/default/file_exchange/denninger/nyc_taxi-0.0.1-py3-none-any.whl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading tables

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import col
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
import numpy as np
from pyspark.sql import SparkSession

from nyctaxi.config import ProjectConfig


spark = SparkSession.builder.getOrCreate()

# Load configuration
config = ProjectConfig.from_yaml(config_path="project_config.yml")
catalog_name = config.catalog_name
schema_name = config.schema_name
pipeline_id = config.pipeline_id

# Ensure 'Id' column is cast to string in Spark before converting to Pandas
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set_an") \
                 .withColumn("pickup_zip", col("pickup_zip").cast("string")) \
                 .toPandas()

test_set = spark.table(f"{catalog_name}.{schema_name}.test_set") \
                 .withColumn("pickup_zip", col("pickup_zip").cast("string")) \
                 .toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get the most important features using random forest model, NOT NECESSARY BECAUSE I ONLY HAVE ONE FEATURE!

# COMMAND ----------

""" import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Encode categorical and datetime variables
def preprocess_data(df):
    label_encoders = {}
    for col in df.select_dtypes(include=['object', 'datetime']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    return df, label_encoders

train_set, label_encoders = preprocess_data(train_set)

# Define features and target (adjust columns accordingly)
features = train_set.drop(columns=["SalePrice"])
target = train_set["SalePrice"]

# Train a Random Forest model
model = RandomForestRegressor(random_state=42)
model.fit(features, target)

# Identify the most important features
feature_importances = pd.DataFrame({
    'Feature': features.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("Top 5 important features:")
print(feature_importances.head(5)) """



# COMMAND ----------

# MAGIC %md
# MAGIC #### Generate 2 synthetic datasets, similar distribution to the existing data and skewed

# COMMAND ----------

from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype, is_object_dtype
import numpy as np
import pandas as pd

def create_synthetic_data(df, drift=False, num_rows=100):
    synthetic_data = pd.DataFrame()

    for column in df.columns:
        dtype = df[column].dtype

        if is_numeric_dtype(dtype) and column != 'pickup_zip':
            mean, std = df[column].mean(), df[column].std()
            synthetic_data[column] = np.random.normal(mean, std, num_rows)
            synthetic_data[column] = synthetic_data[column].astype(dtype)  # Ensure correct data type

        elif is_object_dtype(dtype) or str(dtype).startswith("category"):
            # Generate categorical values based on the distribution in the source data
            synthetic_data[column] = np.random.choice(df[column].unique(), num_rows, 
                                                      p=df[column].value_counts(normalize=True))

        elif is_datetime64_any_dtype(dtype):
            # Generate random dates within the range of existing values
            min_date, max_date = df[column].min(), df[column].max()
            if min_date < max_date:
                synthetic_data[column] = pd.to_datetime(
                    np.random.randint(min_date.value, max_date.value, num_rows)
                )
            else:
                synthetic_data[column] = [min_date] * num_rows

        else:
            # For other types, replicate random samples from existing data
            synthetic_data[column] = np.random.choice(df[column], num_rows)

    # # Generate unique IDs
    # new_ids = []
    # i = max(existing_ids) + 1 if existing_ids else 1
    # while len(new_ids) < num_rows:
    #     if i not in existing_ids:
    #         new_ids.append(i)
    #     i += 1
    # synthetic_data['Id'] = new_ids

    # existing_ids.update(new_ids)  # Add this line to update the existing IDs set

    if drift:
        # Skew the top features to introduce drift
        top_features = ["trip_distance"]  # Select top 2 features
        for feature in top_features:
            if feature in synthetic_data.columns:
                synthetic_data[feature] = synthetic_data[feature] * 1.5

    return synthetic_data

# Generate and visualize fake data

combined_set = pd.concat([train_set, test_set], ignore_index=True)
existing_ids = set(int(id) for id in combined_set['Id'])

synthetic_data_normal = create_synthetic_data(train_set_an,  drift=False, num_rows=200)
synthetic_data_skewed = create_synthetic_data(train_set_an, drift=True, num_rows=200)

print(synthetic_data_normal.dtypes)
print(synthetic_data_normal.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Add timestamp

# COMMAND ----------

synthetic_normal_df = spark.createDataFrame(synthetic_data_normal)
synthetic_normal_df_with_ts = synthetic_normal_df.withColumn(
    "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
)

synthetic_normal_df_with_ts.write.mode("append").saveAsTable(
    f"{catalog_name}.{schema_name}.inference_set_normal_an"
)

synthetic_skewed_df = spark.createDataFrame(synthetic_data_skewed)
synthetic_skewed_df_with_ts = synthetic_skewed_df.withColumn(
    "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
)

synthetic_skewed_df_with_ts.write.mode("append").saveAsTable(
    f"{catalog_name}.{schema_name}.inference_set_skewed_an"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to feature table

# COMMAND ----------

import time
from databricks.sdk import WorkspaceClient

workspace = WorkspaceClient()

#write into feature table; update online table
spark.sql(f"""
    INSERT INTO {catalog_name}.{schema_name}.features_an
    SELECT pickup_zip, trip_distance
    FROM {catalog_name}.{schema_name}.inference_set_normal_an
""")

#write into feature table; update online table
spark.sql(f"""
    INSERT INTO {catalog_name}.{schema_name}.features_an
    SELECT pickup_zip, trip_distance
    FROM {catalog_name}.{schema_name}.inference_set_skewed_an
""")
  
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