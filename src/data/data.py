import pandas as pd
import numpy as np

from features.features import add_features

from utils.data import save_processed_data, load_processed_data
from utils.data import save_glove_matrix, load_glove_matrix
from utils.data import save_WTI, load_WTI
from .data_preprocessing import data_preprocessing
from .data_reader import data_reader, glove_reader
from .glove_reader import glove_embedding
from utils.memory import reduce_mem_usage
###


def __df_column_to_numpy(df_column: pd.Series) -> np.ndarray:
    return np.array([np.array(xi) for xi in df_column.to_numpy()])


def __cast_to_numpy_float(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float)


def __data_to_numpy(df: pd.DataFrame):
    # question_index = np.array(df["question_index"].to_numpy())
    # passage = df["word_index_passage_padded"].to_numpy()
    # question = df["word_index_question_padded"].to_numpy()
    # label = df["label_padded"].to_numpy()
    # pos = df["pos_onehot_padded"].to_numpy()
    # ner = df["ner_onehot_padded"].to_numpy()
    # tf = df["term_frequency_padded"].to_numpy()
    # exact_match = df["exact_match_padded"].to_numpy()
    # output = (passage, question, label, pos, ner, tf, exact_match)
    # for param in output:
    #     for i, p in enumerate(param):
    #         param[i] = np.array(p)

    tf = __df_column_to_numpy(df["term_frequency_padded"])
    pos = __df_column_to_numpy(df["pos_onehot_padded"])
    ner = __df_column_to_numpy(df["ner_onehot_padded"])
    label = __df_column_to_numpy(df["label_padded"])
    passage = __df_column_to_numpy(df["word_index_passage_padded"])
    question = __df_column_to_numpy(df["word_index_question_padded"])
    exact_match = __df_column_to_numpy(df["exact_match_padded"])
    question_index = __df_column_to_numpy(df["question_index"])

    tf = __cast_to_numpy_float(tf)
    pos = __cast_to_numpy_float(pos)
    ner = __cast_to_numpy_float(ner)
    label = __cast_to_numpy_float(label)
    passage = __cast_to_numpy_float(passage)
    question = __cast_to_numpy_float(question)
    exact_match = __cast_to_numpy_float(exact_match)
    question_index = __cast_to_numpy_float(question_index)

    return question_index, passage, question, label, pos, ner, tf, exact_match


def __export_df(df, OHE_pos, OHE_ner, glove_dim):
    cols = [
        "title", "answer", "word_tokens_passage_padded", "word_tokens_question_padded",
        "pos_padded", "ner_padded"
    ]

    df.drop(cols, inplace=True, axis=1)
    df = df.reset_index(drop=True)

    save_processed_data(df, OHE_pos, OHE_ner, glove_dim)

#
# def data_to_numpy(df: pd.DataFrame):
#     question_index = np.array(df["question_index"].to_numpy())
#     passage = df["word_index_passage_padded"].to_numpy()
#     question = df["word_index_question_padded"].to_numpy()
#     label = df["label_padded"].to_numpy()
#     pos = df["pos_onehot_padded"].to_numpy()
#     ner = df["ner_onehot_padded"].to_numpy()
#     tf = df["term_frequency_padded"].to_numpy()
#     exact_match = df["exact_match_padded"].to_numpy()
#
#     # question_index = df["question_index"]
#     # passage = df["word_index_passage_padded"]
#     # question = df["word_index_question_padded"]
#     # label = df["label_padded"]
#     # pos = df["pos_onehot_padded"]
#     # ner = df["ner_onehot_padded"]
#     # tf = df["term_frequency_padded"]
#     # exact_match = df["exact_match_padded"]
#     output = (passage, question, label, pos, ner, tf, exact_match)
#     for param in output:
#         for i, p in enumerate(param):
#             param[i] = np.array(p)
#     return question_index, passage, question, label, pos, ner, tf, exact_match


def get_data(glove_dim, debug=False):
    print("Trying getting glove")
    glove_matrix = load_glove_matrix(glove_dim)
    WTI = load_WTI(glove_dim)
    print("Trying getting data")
    df = load_processed_data(WTI, glove_dim)

    if glove_matrix is None or WTI is None:
        glove = glove_reader(glove_dim)
        glove_matrix, WTI = glove_embedding(glove, glove_dim)
        save_glove_matrix(glove_matrix, glove_dim)
        save_WTI(WTI, glove_dim)

    if df is None:
        if debug:
            df = data_reader()[0:10].copy()
        else:
            df = data_reader()
        df, _ = reduce_mem_usage(df)
        df = data_preprocessing(df, WTI)
        df, _ = reduce_mem_usage(df)
        df, OHE_pos, OHE_ner = add_features(df, WTI)
        df, _ = reduce_mem_usage(df)
        print("Exporting pkl")
        __export_df(df, OHE_pos, OHE_ner, glove_dim)
        print("Exported pkl")
    df_np = __data_to_numpy(df)
    return df, df_np, glove_matrix, WTI
