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


def data_preprocessing(df: pd.DataFrame, WTI):
    df = data_cleaning(df)
    df = apply_word_index(df, WTI)
    # print("data_preprocessing ended")
    return df

