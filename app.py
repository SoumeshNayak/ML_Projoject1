from src.mlopsproject.loggers import logging
from src.mlopsproject.components.data_ingestion import DataIngestion
from src.mlopsproject.components.data_transformation import DataTransormation
from src.mlopsproject.components.model_trainer import ModelTrainer,ModelTrainerConfig
if __name__=="__main__":
    try:
        #Data ingestion
        data_ing=DataIngestion()
        train_data_path,test_data_path=data_ing.initiate_data_ingestion()
        #Data Transformation
        data_Transformation=DataTransormation()
        train_arr,test_arr,=data_Transformation.initialte_data_transformation(train_data_path,test_data_path)
        #Model Training
        model_trainer=ModelTrainer()
        r2_score=model_trainer.initiate_model_trainer(train_arr,test_arr)
        print(r2_score)
        
    except Exception as e:
        raise e