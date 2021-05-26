import numpy as np
import pandas as pd

from modules.cleaners.fuzzy_matcher import FuzzyMatcher

class CategoricalHelper():
    def __init__(self):
        self.fuzzy_matcher = FuzzyMatcher()

    def get_cleaned_df(self, df, features):
        df.fillna('unknown', inplace=True) 
        self.apply_fuzzy_matching(df)
        self.replace_unknowns(df)

    def apply_fuzzy_matching(self, df):
        # doubles
        self.fuzzy_matcher.clean_orgs(df, ['funder', 'installer'])
        
        # singles
        cols = [
            'region',
            'lga',
            'basin',
            'subvillage',
            'ward',
            'wpt_name'
        ]
        # list(map(lambda x: self.fuzzy_matcher.clean_orgs(df, [x]), cols))

    def replace_unknowns(self, df):
        replace = ['na', 'nan', 'none', 'n/a', '-', '']
        list(map(lambda x: df.replace(x, 'unknown', inplace=True), replace))