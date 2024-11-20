# Databricks notebook source
"""
This script trains a LightGBM model for NYC Taxi Dataset with feature engineering.
Key functionality:
- Loads training and test data from Databricks tables
- Performs feature engineering using Databricks Feature Store
- Creates a pipeline with preprocessing and LightGBM regressor
- Tracks the experiment using MLflow
- Logs model metrics, parameters and artifacts
- Handles feature lookups and custom feature functions
- Outputs model URI for downstream tasks

"""

from databricks import feature_engineering
from pyspark.sql import SparkSession
from databricks.sdk import WorkspaceClient
import mlflow
import argparse
from pyspark.sql import functions as F
from lightgbm import LGBMRegressor
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from nyctaxi.config import ProjectConfig
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)
parser.add_argument(
    "--git_sha",
    action="store",
    default=None,
    type=str,
    required=True,
)
parser.add_argument(
    "--job_run_id",
    action="store",
    default=None,
    type=str,
    required=True,
)

args = parser.parse_args()
root_path = args.root_path
git_sha = args.git_sha
job_run_id = args.job_run_id

#config_path = (f"{root_path}/project_config.yml")
config_path = (f"project_config.yml")
# config_path = ("/Volumes/mlops_test/house_prices/data/project_config.yml")
config = ProjectConfig.from_yaml(config_path=config_path)

# Initialize the Databricks session and clients
spark = SparkSession.builder.getOrCreate()
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")


# COMMAND ----------

# Extract configuration details
num_features = config.num_features
cat_features = config.cat_features
target = config.target
parameters = config.parameters
catalog_name = config.catalog_name
schema_name = config.schema_name


# Define table names and function name
feature_table_name = f"{catalog_name}.{schema_name}.features_an"
function_name = f"{catalog_name}.{schema_name}.calculate_travel_time"

# Load training and test sets
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set_an").drop("trip_distance")
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set_an").toPandas()

# COMMAND ----------

train_set = train_set.sample(fraction=0.1, seed=42)
test_set = test_set.sample(frac=0.1, random_state=42)

# COMMAND ----------

# Feature engineering setup
training_set = fe.create_training_set(
    df=train_set,
    label=target,
    feature_lookups=[
        FeatureLookup(
            table_name=feature_table_name,
            feature_names=["trip_distance"],
            lookup_key="pickup_zip",
        ),
        FeatureFunction(
            udf_name=function_name,
            output_name="travel_time",
            input_bindings={"tpep_pickup_datetime": "tpep_pickup_datetime", "tpep_dropoff_datetime": "tpep_dropoff_datetime"},
        ),
    ],
    exclude_columns=["update_timestamp_utc"]
)


# COMMAND ----------

# Load feature-engineered DataFrame
training_df = training_set.load_df().toPandas()

# Calculate travel time for test set
test_set["travel_time"] = (pd.to_datetime(test_set["tpep_dropoff_datetime"]) - pd.to_datetime(test_set["tpep_pickup_datetime"])).dt.total_seconds() / 60



# COMMAND ----------

# Split features and target
X_train = training_df[num_features + ["travel_time"]]
y_train = training_df[target]
X_test = test_set[num_features + ["travel_time"]]
y_test = test_set[target]

# COMMAND ----------

# Setup preprocessing and model pipeline
#preprocessor = ColumnTransformer(
   # transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)], remainder="passthrough"
#)
pipeline = Pipeline(
    steps=[("regressor", LGBMRegressor(**parameters))]
)

# COMMAND ----------

mlflow.set_experiment(experiment_name="/Shared/mlops_course_annika-fe")

with mlflow.start_run(tags={"branch": "week5",
                            "git_sha": f"{git_sha}",
                            "job_run_id": job_run_id}) as run:
    run_id = run.info.run_id
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Calculate and print metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R2 Score: {r2}")

      # Log model parameters, metrics, and model
    mlflow.log_param("model_type", "LightGBM with preprocessing")
    mlflow.log_params(parameters)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2_score", r2)
    signature = infer_signature(model_input=X_train, model_output=y_pred)

    # Log model with feature engineering
    fe.log_model(
        model=pipeline,
        flavor=mlflow.sklearn,
        artifact_path="lightgbm-pipeline-model-fe",
        training_set=training_set,
        signature=signature,
    )

# COMMAND ----------


model_uri=f'runs:/{run_id}/lightgbm-pipeline-model-fe'
dbutils.jobs.taskValues.set(key="new_model_uri", value=model_uri)
