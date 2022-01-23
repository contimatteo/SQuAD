from data_cleaning import data_cleaning
from data_reader import data_reader
import pandas as pd


def data_preprocessing(*_):
    df = data_reader() 
    df = data_cleaning(df)
    #print("data_preprocessing ended")
    return df


def main():
    pd.set_option('display.max_columns', None)    
    pd.set_option('display.max_colwidth', None)
    print(data_preprocessing()[0:1]) 


if __name__ == "__main__":
    main()
  