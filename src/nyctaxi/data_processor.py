
#from pyspark.sql import spark
from databricks.connect import DatabricksSession
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pyspark.sql import functions as F


#spark = spark.builder.appName("DataProcessor").getOrCreate()
spark = DatabricksSession.builder.profile("adb-6130442328907134").getOrCreate()

class DataProcessor:
    def __init__(self, tablepath, config):
        self.df = self.load_data(tablepath)
        print(self.df.head())
        self.config = config
        self.preprocessor = None

    def load_data(self, tablepath) -> pd.DataFrame:
        """Load data from Spark table into pandas DataFrame.
        
        Args:
            tablepath: path where to find the table
            
        Returns:
            pd.DataFrame: Loaded data
        """
        return spark.sql(f"SELECT * FROM {tablepath}").toPandas()

    def preprocess_data(self):
        """Processes numerical and categorical data"""
        # Remove rows with missing target
        target = self.config["target"]
        self.df = self.df.dropna(subset=[target])

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
    
    def split_data(self, test_size: float = 0.2, random_state: int = 42):
        """Split data into training and testing sets.
        
        Args:
            test_size: Proportion of dataset to include in the test split
            random_state: Random seed for reproducibility
        """ 
        train_set, test_set = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return train_set, test_set
    
    
    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame, spark):
        """Save the train and test sets into Databricks tables.
        
        Args:
            train_set: training data
            test_set: test data
        """

        train_set_with_timestamp = spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", F.to_utc_timestamp(F.current_timestamp(), "UTC"))   
        
        test_set_with_timestamp = spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", F.to_utc_timestamp(F.current_timestamp(), "UTC"))

        train_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config['catalog_name']}.{self.config['schema_name']}.train_set_an")
        
        test_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config['catalog_name']}.{self.config['schema_name']}.test_set_an")

        spark.sql(f"ALTER TABLE {self.config['catalog_name']}.{self.config['schema_name']}.train_set_an SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")
        
        spark.sql(f"ALTER TABLE {self.config['catalog_name']}.{self.config['schema_name']}.test_set_an SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")