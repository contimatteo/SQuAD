from data_cleaning import data_cleaning
from data_reader import load_training_set
import pandas as pd


def data_preprocessing(*_):
    df = load_training_set() 
    print("Data cleaning")
    df = data_cleaning(df)
    print("Data cleaned \n")
    return df

pd.set_option('display.max_columns', None)    
pd.set_option('display.max_colwidth', None)
print(data_preprocessing()[0:1])