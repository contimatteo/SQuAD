from typing import List

from data_utils import load_processed_data
import sys
import os
import pandas as pd
import numpy as np
from data_utils import save_processed_data, load_processed_data, save_WTI, save_glove_matrix, load_WTI, load_glove_matrix
from data_preprocessing import data_preprocessing
from data_reader import data_reader, glove_reader
from glove_reader import glove_embedding
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'features'))
from features import add_features

from ast import literal_eval


def data_to_numpy(df: pd.DataFrame):
    question_index = df["question_index"].to_numpy()
    passage = df["word_index_passage_padded"].to_numpy()
    question = df["word_index_question_padded"].to_numpy()
    label = df["label_padded"].to_numpy()
    pos = df["pos_onehot_padded"].to_numpy()
    ner = df["ner_onehot_padded"].to_numpy()
    tf = df["term_frequency_padded"].to_numpy()
    exact_match = df["exact_match_padded"].to_numpy()

    return question_index, passage, question, label, pos, ner, tf, exact_match


def get_data(glove_dim, debug=False):
    print("Trying getting data")
    df = load_processed_data(glove_dim)
    print("Trying getting glove")
    glove_matrix = load_glove_matrix(glove_dim)
    WTI = load_WTI(glove_dim)

    if glove_matrix is None or WTI is None:
        glove = glove_reader(glove_dim)
        glove_matrix, WTI = glove_embedding(glove, glove_dim)
        save_glove_matrix(glove_matrix, glove_dim)
        save_WTI(WTI, glove_dim)

    if df is None:
        if debug:
            df = data_reader()[0: 5].copy()
        else:
            df = data_reader()
        df = data_preprocessing(df, WTI)
        df = add_features(df, WTI)
        print("Exporting csv")
        save_processed_data(df.values, glove_dim)
        print("Exported csv")
    df_np = data_to_numpy(df)
    return df, df_np, glove_matrix, WTI


def main():

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    df, df_np, glove_matrix, WTI = get_data(50, debug=True)
    df['word_index_passage_padded'] = df['word_index_passage_padded'].apply(literal_eval)
    print(df.columns)
    print(df[0:4])
    print("\n---------------\n")
    question_index, passage, question, label, pos, ner, tf, exact_match = df_np
    print(f"QUESTION INDEX\n{question_index[0:1]}\nPASSAGE\n{passage[0:1]}\nQUESTION\n{question[0:1]}\nLABEL\n{label[0:1]}\nPOS TAGGING\n{pos[0:1]}\nNAME ENTITY RECOGNITION\n{ner[0:1]}\nTERM FREQUENCY\n{tf[0:1]}\nEXACT MATCH\n{exact_match[0:1]}")


if __name__ == "__main__":
    main()




