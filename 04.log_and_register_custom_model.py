# Databricks notebook source
import mlflow
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from mlflow.models import infer_signature
from nyctaxi.config import ProjectConfig
import json
from mlflow import MlflowClient
from mlflow.utils.environment import _mlflow_conda_env
from nyctaxi.utils import adjust_predictions


#mlflow.set_tracking_uri("databricks")
mlflow.set_tracking_uri("databricks://adb-6130442328907134")

#mlflow.set_registry_uri('databricks-uc') 
mlflow.set_registry_uri('databricks-uc://adb-6130442328907134') # It must be -uc for registering models to Unity Catalog
client = MlflowClient()

# COMMAND ----------

config = ProjectConfig.from_yaml(config_path="project_config.yml")


# Extract configuration details
num_features = config.num_features
cat_features = config.cat_features
target = config.target
parameters = config.parameters
catalog_name = config.catalog_name
schema_name = config.schema_name
mlflow_experiment_name = config.mlflow_experiment_name

spark = SparkSession.builder.getOrCreate()


# COMMAND ----------

#run_id = mlflow.search_runs(
    #experiment_names=mlflow_experiment_name,
    #filter_string="tags.branch='week2'",
#).run_id[1]


run_id="17b1f52f351c4b46b77451f51af3490e"
model = mlflow.sklearn.load_model(f'runs:/{run_id}/lightgbm-pipeline-model')
print("model loaded")

# COMMAND ----------

class NYCTaxiWrapper(mlflow.pyfunc.PythonModel):
    
    def __init__(self, model):
        self.model = model
        
    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            predictions = self.model.predict(model_input)
            predictions = {"Prediction": adjust_predictions(
                predictions[0])}
            return predictions
        else:
            raise ValueError("Input must be a pandas DataFrame.")

# COMMAND ----------
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set_an")
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set_an")

X_train = train_set[num_features].toPandas()
y_train = train_set[[target]].toPandas()

X_test = test_set[num_features].toPandas()
y_test = test_set[[target]].toPandas()

# COMMAND ----------
wrapped_model = NYCTaxiWrapper(model) # we pass the loaded model to the wrapper
example_input = X_test.iloc[0:1]  # Select the first row for prediction as example
example_prediction = wrapped_model.predict(context=None, model_input=example_input)
print("Example Prediction:", example_prediction)

# COMMAND ----------
# this is a trick with custom packages
# https://docs.databricks.com/en/machine-learning/model-serving/private-libraries-model-serving.html
# but does not work with pyspark, so we have a better option :-)

mlflow.set_experiment(experiment_name=mlflow_experiment_name)
git_sha = "blub"

with mlflow.start_run(tags={"branch": "week2",
                            "git_sha": f"{git_sha}"}) as run:
    
    run_id = run.info.run_id
    signature = infer_signature(model_input=X_train, model_output={'Prediction': example_prediction})
    dataset = mlflow.data.from_spark(
        train_set, table_name=f"{catalog_name}.{schema_name}.train_set_an", version="0")
    mlflow.log_input(dataset, context="training")
    conda_env = _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=["code/nyc_taxi-0.0.1-py3-none-any.whl", 
                             ],
        additional_conda_channels=None,
    )
    mlflow.pyfunc.log_model(
        python_model=wrapped_model,
        artifact_path="lightgbm-pipeline-model",
        code_paths = ["nyc_taxi-0.0.1-py3-none-any.whl"],
        signature=signature
    )
print("model logged")

# COMMAND ----------
loaded_model = mlflow.pyfunc.load_model(f'runs:/{run_id}/lightgbm-pipeline-model')
loaded_model.unwrap_python_model()

# COMMAND ----------
model_name = f"{catalog_name}.{schema_name}.nyctaxi_model_pyfunc"

model_version = mlflow.register_model(
    model_uri=f'runs:/{run_id}/lightgbm-pipeline-model',
    name=model_name,
    tags={"git_sha": f"{git_sha}"})
# COMMAND ----------

with open("model_version.json", "w") as json_file:
    json.dump(model_version.__dict__, json_file, indent=4)

# COMMAND ----------
model_version_alias = "the_best_model"
client.set_registered_model_alias(model_name, model_version_alias, "1")  
 
model_uri = f"models:/{model_name}@{model_version_alias}"
model = mlflow.pyfunc.load_model(model_uri)

# COMMAND ----------
client.get_model_version_by_alias(model_name, model_version_alias)
# COMMAND ----------
model

print("DONE")