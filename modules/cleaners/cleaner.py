import pandas as pd
import datetime as dt

from modules.cleaners.categorical_helper import CategoricalHelper
from modules.cleaners.continuous_helper import ContinuousHelper

class Cleaner():
    def __init__(self):
        self.cont_helper = ContinuousHelper()
        self.cat_helper = CategoricalHelper()
        
    def get_cleaned_df(self, df):
        self.general_cleaning(df)
        self.cat_helper.get_cleaned_df(df, self.get_cat_features(df))
        self.cont_helper.get_cleaned_df(df, self.get_cont_features(df))
        return df
    
    def general_cleaning(self, df):
        self.convert_to_date(df, ['date_recorded'])
        self.drop_cols(df, ['scheme_name', 'date_recorded']) # why dropping date recorded?
        
    def convert_to_date(self, df, cols):
        for col in cols:
            df[col] = pd.DatetimeIndex(pd.to_datetime(df[col]))
            df[col] = df[col].map(dt.datetime.toordinal)
    
    def drop_cols(self, df, cols):
        for col in cols:
            if col in df.columns:
                df.drop([col], axis=1, inplace=True)
                
    def get_cat_features(self, df):
        self.change_type(df, 'region_code', str)
        self.change_type(df, 'district_code', str)
        self.change_type(df, 'num_private', str)
        return df.select_dtypes(include=['object']).columns
    
    def change_type(self, df, col, new_type):
        df[col] = df[col].astype(new_type)
        
    def get_cont_features(self, df):
        return df.select_dtypes(exclude=['object']).columns
    
        