import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

class Splitter:
    def __init__(self):
        self.split_dfs = {'X_train': None, 'X_test': None, 'y_train': None, 'y_test': None}
        
    def get_splits(self, df, outcome):
        self.set_splits(df, outcome)
        return self.split_dfs
    
    def set_splits(self, df, outcome):
        preds = [i for i in df.columns if i != outcome]
        X = df[preds]
        y = df[[outcome]]
        
        self.split_dfs['X_train'], self.split_dfs['X_test'], self.split_dfs['y_train'], self.split_dfs['y_test'] =\
            train_test_split(X, y, test_size = 0.2, random_state = 42)
        
        list(map(lambda key: self.apply_transformations(key), ['X_train', 'X_test']))
        self.match_columns()
            
            
    def apply_transformations(self, key):
        print(key, ': in')
        
        df = self.split_dfs[key]
        features = self.get_feature_types(df)
        
        df_cat = self.transform_cat(df[features['cat']])
        df_cont = self.transform_cont(df[features['cont']])
                
        merged_df = pd.concat([df_cat, df_cont], axis=1)
        self.split_dfs[key] = merged_df
        
        print(key, ': out')

    def get_feature_types(self, df):
        cat_features = df.select_dtypes(include=['object']).columns
        cont_features = df.select_dtypes(exclude=['object']).columns
        return {'cat': cat_features, 'cont': cont_features}
        
    def transform_cont(self, df):
        df_log = np.log(df)
        df_log.columns = [f'{column}_log' for column in df.columns]
        return df_log.apply(self.normalize)
    
    def normalize(self, feature):
        return (feature - feature.mean()) / feature.std()

    def transform_cat(self, df_cat):
        df_cat = df_cat.astype(str)
        id_index = df_cat.index
        ohe = OneHotEncoder(handle_unknown='ignore')
        ohe.fit(df_cat)
        
        matrix_ohe = ohe.transform(df_cat).toarray()
        df_ohe = pd.DataFrame(matrix_ohe, columns=ohe.get_feature_names(df_cat.columns))
        df_ohe['id'] = id_index
        df_ohe.set_index('id', drop=True, inplace=True)
        return df_ohe
    
    def match_columns(self):
        train_cols = self.split_dfs['X_train'].columns.values.tolist()
        test_cols = self.split_dfs['X_test'].columns.values.tolist()
        self.add_match_cols(train_cols, test_cols, 'X_test')
        self.add_match_cols(test_cols, train_cols, 'X_train')
        
    def add_match_cols(self, source_cols, target_cols, target_df_key):
        for col in source_cols:
            if col not in target_cols:
                self.split_dfs[target_df_key][col] = 0
        

