
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
import sagemaker
import boto3
import re
import os

from modules.constants.role import role
from modules.helpers.data_loader import DataLoader
from modules.helpers.viz_helper import VizHelper
from modules.helpers.pre_processor import PreProcessor
from modules.managers.splits_manager import SplitsManager
from modules.managers.report_manager import ReportManager
from modules.helpers.cleaner import Cleaner

class DataManager:
    def __init__(self, sample_size, run_type_dev=True):

        self.sample_size = sample_size
        self.outcome = 'status_group'
        self.splits = SplitsManager()
        self.process_data(run_type_dev)

    def process_data(self, run_type_dev):
        raw_df = DataLoader().load(self.outcome, run_type_dev, self.sample_size)
        df = Cleaner().clean_df(raw_df)
        # VizHelper().show_visualizations(df, self.outcome)
        self.split_data(df, self.outcome)
   
    def split_data(self, df, outcome):
        X = df[[i for i in df.columns if i != outcome]]
        y = df[[outcome]]
        
        self.splits.X_train, self.splits.X_test,\
            self.splits.y_train, self.splits.y_test =\
            tts(X, y, test_size = 0.2, random_state = 42)

    def get_report(self):
        preprocessor = PreProcessor().get_preprocessor(self.splits.X_train)
        ReportManager(self.outcome).run_reports(preprocessor, self.splits)
        
