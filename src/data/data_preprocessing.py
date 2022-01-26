from data_cleaning import data_cleaning
from data_reader import data_reader
import pandas as pd
from typing import Any, Tuple
from glove_reader import glove_embedding
from word_to_index import WordToIndex
import numpy as np


def apply_word_index(df: pd.DataFrame, WTI: WordToIndex):
    df["word_index_passage"] = df.apply(
        lambda x: [p for p in WTI.get_list_index(x["word_tokens_passage"])], axis=1)
    df["word_index_question"] = df.apply(
        lambda x: [p for p in WTI.get_list_index(x["word_tokens_question"])], axis=1)
    return df


def data_preprocessing(*_):
    df, glove = data_reader()
    df = data_cleaning(df)
    glove_matrix, WTI = glove_embedding(df, glove)
    df = apply_word_index(df, WTI)
    # print("data_preprocessing ended")
    return df, glove_matrix, WTI


def main():
    pd.set_option('display.max_columns', None)    
    pd.set_option('display.max_colwidth', None)
    df = data_preprocessing()
    print(df.columns)
    print(df[0:1])


if __name__ == "__main__":
    main()
  