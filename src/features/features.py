import pandas as pd

from .extractor import FeatureExtractor
from .padding import apply_padding_to_df
from .word_to_index import WordToIndex

###


class Features:

    @staticmethod
    def __drop_useless_columns(df: pd.DataFrame):
        useless_columns = [
            "word_index_passage", "word_tokens_passage", "word_index_question", "pos", "pos_onehot",
            "ner", "ner_onehot", "exact_match", "term_frequency", "mask_passage", "mask_question"
        ]

        if "label" in df:
            useless_columns.append("label")

        df.drop(useless_columns, axis=1, inplace=True)

        return df

    #

    @staticmethod
    def build(df: pd.DataFrame, wti: WordToIndex):
        print(
            "Applying POS Tagging, name entity recognition(NER), term frequency(TF), lemmatization"
        )
        df, ohe_pos, ohe_ner = FeatureExtractor.from_df(df)
        print(
            "Applied POS Tagging, name entity recognition(NER), term frequency(TF), lemmatization"
        )

        print("Applying Padding")
        df = apply_padding_to_df(df, wti, ohe_pos, ohe_ner)
        df = Features.__drop_useless_columns(df)
        print("Applied Padding")

        return df, ohe_pos, ohe_ner
