import pandas as pd

from features import WordToIndex

from .cleaner import clean_data

###


class DataPreprocessing:

    @staticmethod
    def __apply_word_index(df: pd.DataFrame, wti: WordToIndex):
        df["word_index_passage"] = df.apply(
            lambda x: [p for p in wti.get_list_index(x["word_tokens_passage"])], axis=1
        )
        df["word_index_question"] = df.apply(
            lambda x: [p for p in wti.get_list_index(x["word_tokens_question"])], axis=1
        )
        return df

    @staticmethod
    def apply(df: pd.DataFrame, wti):
        df = clean_data(df)
        df = DataPreprocessing.__apply_word_index(df, wti)
        return df
