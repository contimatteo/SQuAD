import sys
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk import pos_tag
import pandas as pd
import os

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data'))
from data_preprocessing import data_preprocessing
from lemmatize import apply_lemmatize


def apply_pos_tag(df: pd.DataFrame):
    df["pos"] = df.apply(
        lambda x: [p[1] for p in pos_tag(x["word_tokens_passage"])], axis=1)
    return df


def extract_features(df: pd.DataFrame):
    df = apply_pos_tag(df)
    df = apply_lemmatize(df)
    return df


def main_test():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)

    df_cleaned = data_preprocessing()
    print("Applying POS Tagging, name entity recognition(NER), term frequency(TF)")
    df = extract_features(df_cleaned[0:5].copy())
    print("Applied POS Tagging, name entity recognition(NER), term frequency(TF)")
    print(df.columns)

    print(df[0:1])

    # my_fun(arg1 = "pippo", arg2 = "pluto")


def main():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)

    df_cleaned = data_preprocessing()
    print("Applying POS Tagging, name entity recognition(NER), term frequency(TF)")
    df = extract_features(df_cleaned)
    print("Applied POS Tagging, name entity recognition(NER), term frequency(TF)")
    print(df.columns)

    print(df[0:1])


if __name__ == "__main__":
    main_test()
