import numpy as np
import pandas as pd
import datetime as dt

from modules.helpers.fuzzy_matcher import FuzzyMatcher

class Cleaner():
    def __init__(self):
        pass

    def clean_df(self, df):
        self.override_numeric_coding(df)
        self.convert_to_date(df, ['date_recorded'])
        self.drop_cols(df, ['scheme_name', 'date_recorded']) # why dropping date recorded?
        FuzzyMatcher().clean_orgs(df, ['funder', 'installer'])

    def convert_to_date(self, df, cols):
        for col in cols:
            df[col] = pd.DatetimeIndex(pd.to_datetime(df[col]))
            df[col] = df[col].map(dt.datetime.toordinal)
    
    def drop_cols(self, df, cols):
        for col in cols:
            if col in df.columns:
                df.drop([col], axis=1, inplace=True)

    def override_numeric_coding(self, df):
        cols = ['region_code', 'district_code', 'num_private']
        list(map(lambda col: self.change_type(df, col, str), cols))
    
    def change_type(self, df, col, new_type):
        df[col] = df[col].astype(new_type)