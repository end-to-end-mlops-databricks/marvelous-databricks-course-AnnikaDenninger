# Databricks notebook source
# MAGIC %pip install ../housing_price-0.0.1-py3-none-any.whl

# COMMAND ----------

dbutils.library.restartPython() 

# COMMAND ----------
import yaml
from databricks import feature_engineering
from pyspark.sql import SparkSession
from databricks.sdk import WorkspaceClient
import mlflow
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


# Initialize the Databricks session and clients
spark = SparkSession.builder.getOrCreate()
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

# COMMAND ----------

# COMMAND ----------

#mlflow.set_tracking_uri("databricks")
mlflow.set_tracking_uri("databricks://adb-6130442328907134")
#mlflow.set_registry_uri('databricks-uc') 
mlflow.set_registry_uri('databricks-uc://adb-6130442328907134') # It must be -uc for registering models to Unity Catalog


# Load configuration
config = ProjectConfig.from_yaml(config_path="project_config.yml")

# Extract configuration details
num_features = config.num_features
cat_features = config.cat_features
target = config.target
parameters = config.parameters
catalog_name = config.catalog_name
schema_name = config.schema_name
mlflow_experiment_name = config.mlflow_experiment_name

# Define table names and function name
feature_table_name = f"{catalog_name}.{schema_name}.features_an"
function_name = f"{catalog_name}.{schema_name}.calculate_travel_time"


# COMMAND ----------
# Load training and test sets
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set_an")
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set_an")


# COMMAND ----------
# Create or replace the features_an table
spark.sql(f"""
CREATE OR REPLACE TABLE {catalog_name}.{schema_name}.features_an
(pickup_zip STRING NOT NULL,
 trip_distance INT);
""")

spark.sql(f"ALTER TABLE {catalog_name}.{schema_name}.features_an "
          "ADD CONSTRAINT taxitrip_pk PRIMARY KEY(pickup_zip);")

spark.sql(f"ALTER TABLE {catalog_name}.{schema_name}.features_an "
          "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

# Insert data into the feature table from both train and test sets
spark.sql(f"INSERT INTO {catalog_name}.{schema_name}.features_an "
          f"SELECT pickup_zip, trip_distance FROM {catalog_name}.{schema_name}.train_set_an")
spark.sql(f"INSERT INTO {catalog_name}.{schema_name}.features_an "
          f"SELECT pickup_zip, trip_distance FROM {catalog_name}.{schema_name}.test_set_an")

# COMMAND ----------
# Define a function to calculate the length of travel time based on tpep_pickup_datetime and tpep_dropoff_datetime 
spark.sql(f"""
CREATE OR REPLACE FUNCTION {function_name}(tpep_pickup_datetime TIMESTAMP, tpep_dropoff_datetime TIMESTAMP)
RETURNS INT
LANGUAGE PYTHON AS
$$
from pyspark.sql.functions import col, unix_timestamp

time_difference_seconds = (
    unix_timestamp(col("tpep_dropoff_datetime")) - unix_timestamp(col("tpep_pickup_datetime"))
)
return time_difference_seconds / 60
$$
""")

# COMMAND ----------
# Load training and test sets
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set_an").drop("trip_distance")
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set_an").toPandas()

# Cast YearBuilt to int for the function input
#train_set = train_set.withColumn("tpep_pickup_datetime", train_set["tpep_pickup_datetime"])
#train_set = train_set.withColumn("tpep_dropoff_datetime", train_set["tpep_dropoff_datetime"])
#train_set = train_set.withColumn("pickup_zip", train_set["pickup_zip"].cast("string"))

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

# Load feature-engineered DataFrame
training_df = training_set.load_df().toPandas()

# Split features and target
X_train = training_df[num_features]
y_train = training_df[target]
X_test = test_set[num_features]
y_test = test_set[target]

# Setup preprocessing and model pipeline
#preprocessor = ColumnTransformer(
    #transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)], remainder="passthrough"
#)
pipeline = Pipeline(
    steps=[
        #("preprocessor", preprocessor), 
        ("regressor", LGBMRegressor(**parameters))]
)

# Set and start MLflow experiment
mlflow.set_experiment(experiment_name="/Shared/mlops_course_annika-fe")
git_sha = "blub"

with mlflow.start_run(tags={"branch": "week2",
                            "git_sha": f"{git_sha}"}) as run:
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
mlflow.register_model(
    model_uri=f'runs:/{run_id}/lightgbm-pipeline-model-fe',
    name=f"{catalog_name}.{schema_name}.nyctaxi_model_fe")
    