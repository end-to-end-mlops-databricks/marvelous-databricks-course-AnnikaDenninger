import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataProcessor:
    def __init__(self, spark, filepath, config):
        self.spark = spark
        self.df = self.load_data(filepath)
        self.config = config
        self.preprocessor = None

    def load_data(self, filepath):
        #return pd.read_csv(filepath)
        return self.spark.read.format("delta").load(filepath)

    def preprocess_data(self):
        # Remove rows with missing target
        target = self.config['target']
        self.df = self.df.dropna(subset=[target])
        
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
                ('num', numeric_transformer, self.config['num_features'])
                #('cat', categorical_transformer, self.config['cat_features'])
            ])

    def split_data(self):
        return self.randomSplit([0.8, 0.2], seed=42)
        #return train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

