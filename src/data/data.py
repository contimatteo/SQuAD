from data_utils import load_processed_data
import sys
import os
import pandas as pd
from data_utils import save_processed_data, load_processed_data
from data_preprocessing import data_preprocessing
from data_reader import data_reader, glove_reader
from glove_reader import glove_embedding
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'features'))
from features import add_features


def get_data(glove_dim, debug=False):
    df = load_processed_data()
    glove = glove_reader(glove_dim)
    glove_matrix, WTI = glove_embedding(glove, glove_dim)
    if df is None:
        if debug:
            df = data_reader()[0: 5].copy()
        else:
            df = data_reader()
        df = data_preprocessing(df, WTI)
        df = add_features(df, WTI)
        print("Exporting csv")
        save_processed_data(df)
        print("Exported csv")
    return df, glove_matrix, WTI


def main():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    df, glove_matrix, WTI = get_data(50, debug=True)
    print(df.columns)
    print(df[0:4])


if __name__ == "__main__":
    main()




