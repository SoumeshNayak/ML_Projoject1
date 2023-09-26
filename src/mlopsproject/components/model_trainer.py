import os
import sys
from dataclasses import dataclass


from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from src.mlopsproject.loggers import logging
from src.mlopsproject.utils import save_object
from src.mlopsproject.utils import evaluate_models

import mlflow
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from urllib.parse import urlparse
@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def eval_metrics(self,actual,pred):
        rmse=np.sqrt(mean_squared_error(actual,pred))
        mae=mean_absolute_error(actual,pred)
        r2=r2_score(actual,pred)
        return rmse,mae,r2   
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Spliting training and test input data")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]               
            )
            models = {
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params={
                "Decision Tree":{
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                },
                "Random Forest":{
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradian Boosting":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'n_estimators': [8,16,32,64,128,256],
                }        
            }
            model_report:dict=evaluate_models(X_train=x_train,y_train=y_train,X_test=x_test,y_test=y_test,
                                             models=models,param=params)
            best_model_score = max(sorted(model_report.values()))
            
            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            print(best_model)
            
            # Convert the params keys in list format
            model_names=list(params.keys())
            
            actual_model=""
            for model in model_names:
                if best_model_name==model:
                    actual_model=actual_model+model
            best_params=params[actual_model]  
            
            # Mlflow pipeline
            
            mlflow.set_registry_uri("https://dagshub.com/SoumeshNayak/ML_Projoject1.mlflow")
            tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme
            
            with mlflow.start_run():
                pred_qualit=best_model.predict(x_test)
                
                (rmse,mae,r2)=self.eval_metrics(y_test,pred_qualit)
                
                mlflow.log_params(best_params)
                mlflow.log_metric("rmse",rmse)
                mlflow.log_metric("r2",r2)
                mlflow.log_metric("mae",mae)
                
                # Get the url from dagus hub and track
                if tracking_url_type_store !="file":
                    mlflow.sklearn.log_model(best_model,"model",registered_model_name=actual_model)
                else:
                    mlflow.sklearn.log_model(best_model,"model")     
                               
            
            logging.info("Got best model")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(x_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square
            
        except Exception as e:
            raise e       
