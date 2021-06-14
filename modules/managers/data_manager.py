
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
    #     self.role = role
    #     sess = sagemaker.Session()
    #     self.bucket = sess.default_bucket()
    #     self.prefix = "sagemaker"

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
        # self.save_splits()
        
    # def save_splits(self):
    #     print("saving...")
    #     self.save_to_s3(self.splits.X_train, "train", "X_train")
    #     self.save_to_s3(self.splits.X_train, "train", "y_train")
    #     self.save_to_s3(self.splits.X_train, "test", "X_test")
    #     self.save_to_s3(self.splits.X_train, "test", "y_test")
        
        
    # def save_to_s3(self, df, split_type, file_name):
    #     df.to_csv(f"data/{file_name}.csv", header=False, index=False)
    #     boto3.Session().resource('s3').Bucket(self.bucket).\
    #         Object(os.path.join(self.prefix, f"{split_type}/{file_name}"))\
    #         .upload_file(f"data/{file_name}.csv")
         








    # def split_data(self, df):
    #     train_data, validation_data, test_data = np.split(
    #         df.sample(frac=1, random_state=1729),
    #         [int(0.7 * len(df)), int(0.9 * len(df))],
    #     )
    #     print("saving...")
    #     self.save_to_s3(train_data, "train")
    #     self.save_to_s3(validation_data, "validate")
    #     self.save_to_s3(test_data, "test")
    #     print("saved to s3")

    # def save_to_s3(self, df, file):
    #     df.to_csv(f"data/{file}.csv", header=False, index=False)
    #     boto3.Session().resource('s3').Bucket(self.bucket).\
    #         Object(os.path.join(self.prefix, f"{file}/{file}.csv"))\
    #         .upload_file(f"data/{file}.csv")

    def get_report(self):
        preprocessor = PreProcessor().get_preprocessor(self.splits.X_train)
        ReportManager(self.outcome).run_reports(preprocessor, self.splits)
        # ReportManager(self.bucket, self.prefix, self.outcome).run_reports(preprocessor, self.splits)
        
