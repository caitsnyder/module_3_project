import pandas as pd
pd.options.mode.chained_assignment = None

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import RandomOverSampler

class PreProcessor():
    def __init__(self):
        pass

    def get_preprocessor(self, df):
        return {
            'sampler': self.get_resampler(),
            'col': self.get_col_transformer(df)
            }

    def get_resampler(self):
        return RandomOverSampler(random_state=42)
        
    def get_col_transformer(self, df):
        cont_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])

        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        return ColumnTransformer(
            transformers=[
                ('num', cont_transformer, self.get_cont_features(df)),
                ('cat', cat_transformer, self.get_cat_features(df))])

    def get_cat_features(self, df):
        return df.select_dtypes(include=['object']).columns
        
    def get_cont_features(self, df):
       return df.select_dtypes(exclude=['object']).columns
    
        