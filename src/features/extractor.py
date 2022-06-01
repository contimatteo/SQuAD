import pandas as pd

from .exact_match import ExactMatchFeature
from .lemmatizer import Lemmatizer
from .ner import NER
from .term_frequency import TermFrequency
from .pos import POS

###


class FeatureExtractor:

    @staticmethod
    def __drop_useless_columns_from_df(df: pd.DataFrame):
        useless_columns = ["passage", "question", "lemmatized_passage", "lemmatized_question"]
        df.drop(useless_columns, axis=1, inplace=True)
        return df

    #

    @staticmethod
    def from_df(df: pd.DataFrame):
        print("Applying POS")
        df = POS.apply_to(df)
        df, ohe_pos = POS.apply_ohe_to_df(df)

        print("Applying NER")
        df = NER.apply_to_df(df)
        df, ohe_ner = NER.apply_ohe_to_df(df)

        print("Applying TF")
        df = TermFrequency.apply_to_df(df)

        print("Applying LEMMATIZATION")
        df = Lemmatizer.apply_to_df(df)

        print("Applying EXACT MATCH")
        df = ExactMatchFeature.apply_to_df(df)

        df = FeatureExtractor.__drop_useless_columns_from_df(df)

        return df, ohe_pos, ohe_ner
