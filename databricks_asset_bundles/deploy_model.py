"""
This script handles the deployment of a NYC taxi prediction model to a Databricks serving endpoint.
Key functionality:
- Loads project configuration from YAML
- Retrieves the model version from previous task values
- Updates the serving endpoint configuration with:
  - Model registry reference
  - Scale to zero capability
  - Workload sizing
  - Specific model version
The endpoint is configured for feature-engineered model serving with automatic scaling.
"""

import yaml
import argparse
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedEntityInput
from nyctaxi.config import ProjectConfig

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)

args = parser.parse_args()
root_path = args.root_path

config_path = (f"{root_path}/project_config.yml")
# config_path = ("/Volumes/mlops_test/house_prices/data/project_config.yml")
config = ProjectConfig.from_yaml(config_path=config_path)

model_version = dbutils.jobs.taskValues.get(taskKey="evaluate_model", key="model_version")

workspace = WorkspaceClient()

catalog_name = config.catalog_name
schema_name = config.schema_name

workspace.serving_endpoints.update_config_and_wait(
    name="house-prices-model-serving-fe",
    served_entities=[
        ServedEntityInput(
            entity_name=f"{catalog_name}.{schema_name}.nyctaxi-model-serving-fe",
            scale_to_zero_enabled=True,
            workload_size="Small",
            entity_version=model_version,
        )
    ],
)