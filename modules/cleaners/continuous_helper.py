import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
pd.options.mode.chained_assignment = None

class ContinuousHelper():
    def __init__(self):
        self.imputer = SimpleImputer(missing_values = np.nan, 
                        strategy ='median')
        self.scaler = StandardScaler()
    
    def get_cleaned_df(self, df, outcome):
        cont_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])



        imputer = self.imputer.fit(df)
        arr_imputed = imputer.transform(df)
        arr_scaled =  self.scaler.fit_transform(arr_imputed)
        df = pd.DataFrame(imputer.transform(arr_scaled), columns=df.columns, index=df.index)
        print(arr)
        # list(map(lambda col: self.clean_pre_log(df, col), df.columns))
        # self.show_visualizations(df, features, outcome)
        return df
    
    # def clean_zeros(self, df, cols):
    #     for col in cols:
    #         df[col] = df.apply(
    #             lambda row: np.nan if row[col] == 0 else row[col], axis=1)
    
    # # def impute_median(self, df):
    #     imputer = self.imputer.fit(df)
    #     return pd.DataFrame(imputer.transform(df), columns=df.columns, index=df.index)
            
    def clean_pre_log(self, df, col):
        if df[col].min() > 0:
            return

        elif (df[col].sum() == 0):
            df.drop([col],axis=1, inplace=True)

        elif df[col].min() == 0:
            col_non_zero_min = df.loc[df[col] > 0, col].min()
            offset = col_non_zero_min/2
            df[col] = df.apply(
                lambda row: row[col] + offset,
                axis=1)

        elif df[col].min() < 0:
            col_min = abs(df[col].min()) + 1
            df[col] = df.apply(
                lambda row: row[col] + col_min,
                axis=1)

    def show_visualizations(self, df, features, outcome):
        self.generate_heat_map(df, features)
        self.show_outliers(df, features)
        self.show_basic_correlations(df, features, outcome)

    def generate_heat_map(self, df, features):
        plt.figure(figsize=(7, 6))
        sns.heatmap(df[features].corr(), center=0)
        plt.show()


    def show_outliers(self, df, cols):
        fig, axes = plt.subplots(2, 3, figsize=(9, 6))
        axe = axes.ravel()

        for i, xcol in enumerate(cols):
            sns.boxplot(x=df[xcol], ax=axe[i])
        plt.show()

    def show_basic_correlations(self, df, cols, outcome):
        preds = [i for i in cols if i != outcome]
        fig, axes = plt.subplots(2, 3, figsize=(9, 6))
        axe = axes.ravel()

        for i, xcol in enumerate(preds):
            df.plot(kind='scatter', x=xcol, y=outcome, alpha=0.4, color='b', ax=axe[i])
        
        plt.show()
