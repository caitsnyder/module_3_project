import pandas as pd
import numpy as np

from modules.organizers.paths import Paths
from modules.cleaners.cleaner import Cleaner
from modules.analyzers.splitter import Splitter
from modules.analyzers.model_report import ModelReport

class DataKeeper:
    def __init__(self, run_type_dev=False):
        self.run_type_dev = run_type_dev
        self.outcome = 'status_group'
        self.outcome_values = []
        
        self.cleaned_df = None
        self.splits = None
        
        self.paths = Paths()
        self.cleaner = Cleaner()
        self.splitter = Splitter()
        self.report = ModelReport()
        
        self.get_data()

    def get_data(self):
        df = self.load_df()
        self.preprocess(df)
        
    def load_df(self):
        X_train = self.load_data(self.paths.train_values)
        y_train = self.load_data(self.paths.train_labels)
        
        if self.run_type_dev:
            X_train = X_train.iloc[0:50]
            y_train = y_train.iloc[0:50]
            
        df = pd.concat([X_train, y_train], axis=1, join="inner")
        self.outcome_values = np.unique(df[self.outcome])
        return df
    
    def load_data(self, path):
        df = pd.read_csv(path)
        df.set_index('id', inplace=True)
        return df
        
    def preprocess(self, raw_df):
        self.cleaned_df = self.cleaner.get_cleaned_df(raw_df)
        self.splits = self.splitter.get_splits(self.cleaned_df, self.outcome)
        
    def get_report(self):
        self.report.get_reports(self.splits)
    
    