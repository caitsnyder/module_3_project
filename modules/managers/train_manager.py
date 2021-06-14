import pandas as pd
import numpy as np
import sagemaker
import boto3
from IPython.display import display

from modules.constants.role import role
from sagemaker.inputs import TrainingInput

class TrainManager:
    def __init__(self):
        self.role = role
        sess = sagemaker.Session()
        self.bucket = sess.default_bucket()
        self.prefix = "sagemaker"
        self.s3_input_train, self.s3_input_validation = None, None
        self.train_data()
    
    def train_data(self):
        self.load_data()
        self.implement_pipeline()

    def load_data(self):
        self.s3_input_train = self.load_s3_file("train")
        self.s3_input_validation = self.load_s3_file("test")

    def load_s3_file(self, file):
        return TrainingInput(
            s3_data=f"s3://{self.bucket}/{self.prefix}/{file}", content_type="csv"
        )

    def implement_pipeline(self):
        container = self.get_container()
        sess = self.get_session()

        

    def get_container(self):
        return sagemaker.image_uris.retrieve("xgboost", boto3.Session().region_name, "1")
    
    def get_session(self):
        return sagemaker.Session()

