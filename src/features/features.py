import pandas as pd
import sys
import os
from extract_features import extract_features
from padding import apply_padding_to
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data'))
from data_preprocessing import data_preprocessing


def drop_useless_columns(df: pd.DataFrame):
    useless_columns = ["label", "word_tokens_passage", "word_tokens_question", "word_index_passage",
                       "word_index_question", "pos", "pos_onehot", "ner", "ner_onehot", "exact_match", "term_frequency"]
    df.drop(useless_columns, axis=1, inplace=True)
    return df


def get_features_test():
    df_cleaned, glove_matrix, WTI = data_preprocessing()
    print("Applying POS Tagging, name entity recognition(NER), term frequency(TF), lemmatization")
    df, OHE_pos, OHE_ner = extract_features(df_cleaned[0:5].copy())
    print("Applied POS Tagging, name entity recognition(NER), term frequency(TF), lemmatization")
    print("Applying Padding")
    df = apply_padding_to(df, WTI, OHE_pos, OHE_ner)
    df = drop_useless_columns(df)
    print("Applied Padding")
    return df
    # print(len(df["word_tokens_passage"][0:1][0]))


def get_features():
    df_cleaned, glove_matrix, WTI = data_preprocessing()
    print("Applying POS Tagging, name entity recognition(NER), term frequency(TF), lemmatization")
    df, OHE_pos, OHE_ner = extract_features(df_cleaned)
    df = apply_padding_to(df, WTI)
    print("Applied POS Tagging, name entity recognition(NER), term frequency(TF), lemmatization")
    print("Applying Padding")
    df = apply_padding_to(df, WTI, OHE_pos, OHE_ner)
    df = drop_useless_columns(df)
    print("Applied Padding")
    return df


def main():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    df = get_features_test()
    print(df.columns)
    print(df[0:2])


if __name__ == "__main__":
    main()
