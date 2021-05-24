import numpy as np
import pandas as pd

from modules.cleaners.fuzzy_matcher import FuzzyMatcher

class CategoricalHelper():
    def __init__(self):
        self.fuzzy_matcher = FuzzyMatcher()

    def get_cleaned_df(self, df, features):
        df.fillna('unknown', inplace=True) 
        self.fuzzy_matcher.clean_orgs(df, ['funder', 'installer'])

            
            