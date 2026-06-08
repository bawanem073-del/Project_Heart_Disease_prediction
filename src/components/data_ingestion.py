
import os
import sys
from src.exception import CustomException
from src.logger import logging

from dataclasses import dataclass
import pandas as pd




@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts', 'train.csv')
    test_data_path = os.path.join('artifacts', 'test.csv')
    raw_data_path = os.path.join('artifacts', 'data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()


    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")

        try:
            df = pd.read_csv('notebook/data/health_data.csv')
            logging.info("Read the dataset as dataframe")

        
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    # train_data, test_data = obj.initiate_data_ingestion()
