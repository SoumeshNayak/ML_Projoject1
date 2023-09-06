from src.mlopsproject.loggers import logging
from src.mlopsproject.components.data_ingestion import DataIngestion
if __name__=="__main__":
    try:
        data_ing=DataIngestion()
        data_ing.initiate_data_ingestion()
    except Exception as e:
        raise e