import numpy as np
import pandas as pd
from sklearn.utils import resample

from modules.helpers.viz_helper import VizHelper
from modules.helpers.fuzzy_matcher import FuzzyMatcher

class Cleaner():
    def __init__(self):
        self.viz_helper = VizHelper()

    def clean_df(self, df):
        df.drop(['num_private', 'amount_tsh'], axis=1, inplace=True)
        self.convert_to_string(df)
        self.replace_nan(df)
        FuzzyMatcher().set_fuzzy_matches(df, ['funder', 'installer'])
        return df
    def convert_to_string(self, df):
        cols = [
            'region_code', 
            'district_code', 
            'public_meeting', 
            'permit'
        ]
        list(map(lambda col: self.change_type(df, col, str), cols))
    
    def change_type(self, df, col, new_type):
        df[col] = df[col].astype(new_type)

    def replace_nan(self, df):
        self.replace_null_strings(df, "nan")
        self.replace_null_strings(df, "none")
        self.replace_zeros(df, "longitude")
        self.replace_zeros(df, "latitude")
        self.replace_zeros(df, 'construction_year')
        self.replace_zeros(df, 'population')

    def replace_null_strings(self, df, null_str):
        df.replace(to_replace=null_str, value="unknown", inplace=True)

    def replace_zeros(self, df, col):
        df[col] = df.apply(lambda row: np.nan 
            if row[col] == 0 else row[col], axis=1)