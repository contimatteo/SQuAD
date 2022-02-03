import pandas as pd
import numpy as np
from features.word_to_index import WordToIndex

from .data_cleaning import data_cleaning

###


def apply_word_index(df: pd.DataFrame, wti: WordToIndex):
    df["word_index_passage"] = df.apply(
        lambda x: [p for p in wti.get_list_index(x["word_tokens_passage"])], axis=1
    )
    df["word_index_question"] = df.apply(
        lambda x: [p for p in wti.get_list_index(x["word_tokens_question"])], axis=1
    )
    return df


def data_preprocessing(df: pd.DataFrame, wti):
    df = data_cleaning(df)
    df = apply_word_index(df, wti)
    # print("data_preprocessing ended")
    return df
