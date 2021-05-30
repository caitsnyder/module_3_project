from modules.helpers.cleaner import Cleaner
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


from modules.helpers.data_loader import DataLoader
from modules.helpers.viz_helper import VizHelper
from modules.helpers.pre_processor import PreProcessor
from modules.managers.splits_manager import SplitsManager
from modules.managers.report_manager import ReportManager

class DataManager:
    def __init__(self, run_type_dev=False):
        self.outcome = 'status_group'
        self.splits = SplitsManager()
        self.process_data(run_type_dev)

    def process_data(self, run_type_dev):
        raw_df = DataLoader().load(self.outcome, run_type_dev)
        df = Cleaner().clean_df(raw_df)
        # VizHelper().show_visualizations(df, self.outcome)
        self.set_splits(df, self.outcome)
   
    def set_splits(self, df, outcome):
        X = df[[i for i in df.columns if i != outcome]]
        y = df[[outcome]]
        
        self.splits.X_train, self.splits.X_test,\
            self.splits.y_train, self.splits.y_test =\
            train_test_split(X, y, test_size = 0.2, random_state = 42)
            
    def get_report(self):
        preprocessor = PreProcessor().get_preprocessor(self.splits.X_train)
        ReportManager(self.outcome).run_reports(preprocessor, self.splits)
