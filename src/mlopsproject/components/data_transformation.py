import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.mlopsproject.loggers import logging
from src.mlopsproject.utils import save_object
@dataclass
class DataTransormationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')
class DataTransormation:
    def __init__(self):
        self.data_transormation_config=DataTransormationConfig()
    def  get_data_transformer_odject(self):
         try:
              numerical_col=['reading score', 'writing score']
              cat_col=['gender', 'race/ethnicity', 'parental level of education', 'lunch','test preparation course']
              
              num_pipeline=Pipeline(
                  steps=[
                      ("imputer",SimpleImputer(strategy="median")),
                      ("scaler",StandardScaler())
                  ]
              )
              cat_pipeline=Pipeline(
                  steps=[
                      ("imputer",SimpleImputer(strategy="most_frequent")),
                      ("one_hot_encoding",OneHotEncoder()),
                      ("scaler",StandardScaler(with_mean=False)),
                  ]
              )
              logging.info(f"cat columns encoding complited")
              logging.info(f"Num columns scaling complited")
              
              preprocessor=ColumnTransformer(
                  [
                      ("num_pipeline",num_pipeline,numerical_col),
                      ("cat_pipelines",cat_pipeline,cat_col)
                  ]
              )
              return preprocessor     
         except Exception as e:
            raise e
    def initialte_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("Read train and test data complited")
            
            preprocessor_obj=self.get_data_transformer_odject()
            target_column="math score"
            numerical_col=['reading score', 'writing score']
            input_feature_train_df=train_df.drop(columns=[target_column],axis=1)
            target_feature_train_df=train_df[target_column]
            
            input_feature_test_df=test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df=test_df[target_column]
            
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            logging.info(f"Saved preprocessing object.")
            
            save_object(
                file_path=self.data_transormation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            
            return(
                train_arr,
                test_arr,
                # self.data_transormation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise e     
      