import numpy as np
import pandas as pd

class ContinuousHelper():
    def __init__(self):
        pass
    
    def get_cleaned_df(self, df, features):
        self.impute_zeros(df, ['construction_year', 'population'])
        list(map(lambda col: self.handle_zeros_pre_log(df, col), features))
        list(map(lambda col: self.handle_negatives_pre_log(df, col), features))
        
    def impute_zeros(self, df, cols):
        for col in cols:
            df[col] = df.apply(
                lambda row: np.nan if row[col] == 0 else row[col], axis=1)
            self.impute_median(df, col)
    
    def impute_median(self, df, col):
        df_one_col = df[[col]]
        df_one_col.fillna(df_one_col.median(), inplace=True)        
        df[col] = df_one_col[col]
            
    def handle_zeros_pre_log(self, df, col):
        if df[col].min() != 0:
            return
        elif (df[col].min() == 0 and df[col].max() == 0):
            df.drop([col],axis=1, inplace=True)
        else:
            col_non_zero_min = df.loc[df[col] > 0, col].min()
            offset = col_non_zero_min/2
            df[col] = df.apply(
                lambda row: row[col] + offset,
                axis=1)
            
    def handle_negatives_pre_log(self, df, col):
        if df[col].min() >= 0:
            return
        else:
            col_min = abs(df[col].min()) + 1
            df[col] = df.apply(
                lambda row: row[col] + col_min,
                axis=1)
