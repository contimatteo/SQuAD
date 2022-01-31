import pandas as pd

from .extract_features import extract_features
from .padding import apply_padding_to
from .word_to_index import WordToIndex

###


def drop_useless_columns(df: pd.DataFrame):
    useless_columns = [
        "label", "word_tokens_passage", "word_tokens_question", "word_index_passage",
        "word_index_question", "pos", "pos_onehot", "ner", "ner_onehot", "exact_match",
        "term_frequency"
    ]
    df.drop(useless_columns, axis=1, inplace=True)
    return df


def add_features(df: pd.DataFrame, WTI: WordToIndex):
    print("Applying POS Tagging, name entity recognition(NER), term frequency(TF), lemmatization")
    df, OHE_pos, OHE_ner = extract_features(df)
    print("Applied POS Tagging, name entity recognition(NER), term frequency(TF), lemmatization")
    print("Applying Padding")
    df = apply_padding_to(df, WTI, OHE_pos, OHE_ner)
    df = drop_useless_columns(df)
    print("Applied Padding")
    return df
