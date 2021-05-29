import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


from modules.helpers.data_loader import DataLoader
from modules.helpers.pre_processor import PreProcessor
from modules.managers.splits_manager import SplitsManager
from modules.managers.report_manager import ReportManager

class DataManager:
    def __init__(self, run_type_dev=False):
        self.outcome = 'status_group'
        self.split_dfs = SplitsManager()
        self.process_data(run_type_dev)

    def process_data(self, run_type_dev):
        raw_df = DataLoader().load(self.outcome, run_type_dev)
        splits = self.get_splits(raw_df, self.outcome)
        pipe = PreProcessor().get_pipeline(splits.X_train, self.outcome)
        ReportManager().run_reports(pipe, splits)
   
    def get_splits(self, df, outcome):
        X = df[[i for i in df.columns if i != outcome]]
        y = df[[outcome]]
        
        splits = self.split_dfs.X_train, self.split_dfs.X_test,\
                self.split_dfs.y_train, self.split_dfs.y_test =\
            train_test_split(X, y, test_size = 0.2, random_state = 42)
        return splits