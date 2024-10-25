
#from pyspark.sql import SparkSession
from databricks.connect import DatabricksSession
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

#spark = SparkSession.builder.appName("DataProcessor").getOrCreate()
spark = DatabricksSession.builder.profile("adb-6130442328907134").getOrCreate()


class DataProcessor:
    def __init__(self, tablepath, config):
        self.df = self.load_data(tablepath)
        print(self.df.head())
        self.config = config
        self.X = None
        self.y = None
        self.preprocessor = None

    def load_data(self, tablepath):
        return spark.sql(f"SELECT * FROM {tablepath}").toPandas()

    def preprocess_data(self):
        # Remove rows with missing target
        target = self.config["target"]
        self.df = self.df.dropna(subset=[target])

        # Separate features and target
        self.X = self.df[self.config["num_features"]]
        self.y = self.df[target]

        # Create preprocessing steps for numeric and categorical data
        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.config["num_features"]),
                ('cat', categorical_transformer, self.config['cat_features'])
            ]
        )

    def split_data(self, test_size=0.2, random_state=42):
        return train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)