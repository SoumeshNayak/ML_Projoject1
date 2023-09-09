# Database(Mysql)-->data-->train test split
import os
import sys
from src.mlopsproject.loggers import logging

import pandas as pd

from dataclasses import dataclass
from src.mlopsproject.utils import read_sql_data
from sklearn.model_selection import train_test_split
from src.mlopsproject.components.data_transformation import DataTransormation,DataTransormationConfig
@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig() 
    def initiate_data_ingestion(self):
        try:
            df=read_sql_data()
            logging.info("Reading from mysql database")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            logging.info("Doing Train-test-split")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            logging.info("Data ingestion is complited")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise e 
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    data_transformation=DataTransormation()
    data_transformation.initialte_data_transformation(train_data,test_data)
    
    
                 
    
