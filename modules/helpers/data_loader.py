import pandas as pd
import numpy as np

from modules.constants.path_constants import paths

class DataLoader:
    def __init__(self):
        pass

    def load(self, outcome, run_type_dev):
        X_train = self.load_from_path(paths['train_values'])
        y_train = self.load_from_path(paths['train_labels'])
        
        if run_type_dev:
            X_train = X_train.iloc[0:100]
            y_train = y_train.iloc[0:100]
            
        df = pd.concat([X_train, y_train], axis=1, join="inner")
        self.outcome_values = np.unique(df[outcome])
        return df
    
    def load_from_path(self, path):
        df = pd.read_csv(path)
        df.set_index('id', inplace=True)
        return df
        