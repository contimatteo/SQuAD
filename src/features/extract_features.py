import sys
import nltk
from nltk import pos_tag
import pandas as pd
import os
from sklearn import preprocessing
import numpy as np
from one_hot_encoder import OneHotEncoder
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data'))
from data_preprocessing import data_preprocessing
from lemmatize import apply_lemmatize
from name_entity_recognition import apply_ner
from term_frequency import apply_term_frequency


def apply_pos_tag(df: pd.DataFrame):
    df["pos"] = df.apply(
        lambda x: [p[1] for p in pos_tag(x["word_tokens_passage"])], axis=1)
    return df


def extract_features(df: pd.DataFrame):
    df = apply_pos_tag(df)
    df = apply_pos_one_hot(df)
    df = apply_ner(df)
    df = apply_term_frequency(df)
    df = apply_lemmatize(df)
    return df


def apply_pos_one_hot(df: pd.DataFrame):
    pos_list = ['LS', 'TO', 'VBN', "''", 'WP', 'UH', 'VBG', 'JJ', 'VBZ', '--', 'VBP', 'NN', 'DT', 'PRP', ':', 'WP$',
                'NNPS', 'PRP$', 'WDT', '(', ')', '.', ',', '``', '$', 'RB', 'RBR', 'RBS', 'VBD', 'IN', 'FW', 'RP',
                'JJR', 'JJS', 'PDT', 'MD', 'VB', 'WRB', 'NNP', 'EX', 'NNS', 'SYM', 'CC', 'CD', 'POS']
    OHE = OneHotEncoder()
    OHE.fit(pos_list)
    df["pos_onehot"] = df.apply(
        lambda x: [p for p in OHE.transform(x["pos"])], axis=1)
    return df


def main_test():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)

    df_cleaned, glove_matrix, WTI = data_preprocessing()
    print("Applying POS Tagging, name entity recognition(NER), term frequency(TF)")
    df = extract_features(df_cleaned[0:5].copy())
    print("Applied POS Tagging, name entity recognition(NER), term frequency(TF)")
    print(df.columns)
    print(df[0:1])


def main():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)

    df_cleaned, glove_matrix, WTI = data_preprocessing()
    print("Applying POS Tagging, name entity recognition(NER), term frequency(TF)")
    df = extract_features(df_cleaned)
    print("Applied POS Tagging, name entity recognition(NER), term frequency(TF)")
    print(df.columns)
    print(df[0:1])

    # print(nltk.help.upenn_tagset())


if __name__ == "__main__":
    main_test()
