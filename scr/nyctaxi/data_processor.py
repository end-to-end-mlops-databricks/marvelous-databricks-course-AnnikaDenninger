import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pyspark.sql.functions import dayofmonth, month, year, hour

class DataProcessor:
    def __init__(self, spark, filepath, config):
        self.spark = spark
        self.df = self.load_data(filepath)
        self.config = config
        self.X = None
        self.y = None
        self.preprocessor = None

    def load_data(self, filepath):
        #return pd.read_csv(filepath)
        return self.spark.read.format("delta").load(filepath)

    def preprocess_data(self):
        # Remove rows with missing target
        target = self.config['target']
        self.df = self.df.dropna(subset=[target])
        
        # Separate features and target
        self.X = self.df[self.config['num_features'] + self.config['timestamp_features']]
        self.y = self.df[target]

        # Create preprocessing steps for numeric and timestamp data
        timestamp_transformer = Pipeline(steps=[
            ('extract_features', TimestampFeatureExtractor())
            #('imputer', SimpleImputer(strategy='most_frequent'))
            #('scaler', StandardScaler())
        ])
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # categorical_transformer = Pipeline(steps=[
        #     ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        #     ('onehot', OneHotEncoder(handle_unknown='ignore'))
        # ])
        
        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.config['num_features']),
                #('cat', categorical_transformer, self.config['cat_features'])
                ('timestamp', timestamp_transformer, self.config['timestamp_features'])
            ])

    def split_data(self):
        return self.randomSplit([0.8, 0.2], seed=42)
        #return train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

class TimestampFeatureExtractor:
    def transform(self, df):
        df = df.withColumn('tpep_pickup_datetime_hour', hour(df['tpep_pickup_datetime']))
        df = df.withColumn('tpep_dropoff_datetime_hour', hour(df['tpep_dropoff_datetime']))

        # Return the transformed dataframe
        return df.select(
            'tpep_pickup_datetime_hour',
            'tpep_dropoff_datetime_hour'
        )
