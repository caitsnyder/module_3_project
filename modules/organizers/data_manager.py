import pandas as pd
import numpy as np

from modules.cleaners.data_loader import DataLoader
from modules.cleaners.pre_processor import PreProcessor
from modules.analyzers.splitter import Splitter
from modules.analyzers.model_report import ModelReport

class DataManager:
    def __init__(self, run_type_dev=False):
        self.outcome = 'status_group'
        self.process_data(run_type_dev)

    def process_data(self, run_type_dev):
        raw_df = DataLoader().load(self.outcome, run_type_dev)
        splits = Splitter().get_splits(raw_df, self.outcome)
        pipe = PreProcessor().get_pipeline(splits.X_train, self.outcome)
        ModelReport().run_reports(pipe, splits)
   
    def get_report(self):
        self.report.get_reports(self.splits)
    
    