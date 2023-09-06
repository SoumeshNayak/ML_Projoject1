import os
import sys
from src.mlopsproject.loggers import logging
from dotenv import load_dotenv
import pandas as pd
import pymysql
load_dotenv()

host=os.getenv("host")
user=os.getenv("user")
password=os.getenv("password")
db=os.getenv("db")

def read_sql_data():
    logging.info("Reading SQL database started")
    try:
        mydb=pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        logging.info("Connection established",mydb)
        df=pd.read_sql_query('SELECT * FROM students.studentsperformance',mydb)
        print(df.head())
        return df
        
    except Exception as e:
        raise e