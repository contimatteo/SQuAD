import pandas as pd
import numpy as np

from features.features import add_features
from utils.data_storage import save_processed_data, load_processed_data
from utils.data_storage import save_glove_matrix, load_glove_matrix
from utils.data_storage import save_WTI, load_WTI
from utils.memory import reduce_mem_usage

from .data_preprocessing import data_preprocessing
from .data_reader import data_reader, glove_reader
from .data_reader import save_evaluation_data_df, load_evaluation_data_df
from .glove_reader import glove_embedding
from utils.data_storage import create_tmp_directories, load_config_data, save_config_data, clean_all_data_cache

###


def __df_column_to_numpy(df_column: pd.Series) -> np.ndarray:
    return np.array([np.array(xi) for xi in df_column.to_numpy()])


def __cast_to_numpy_float(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float)


def __data_to_numpy(df: pd.DataFrame):
    tf = __df_column_to_numpy(df["term_frequency_padded"])
    pos = __df_column_to_numpy(df["pos_onehot_padded"])
    ner = __df_column_to_numpy(df["ner_onehot_padded"])
    passage = __df_column_to_numpy(df["word_index_passage_padded"])
    question = __df_column_to_numpy(df["word_index_question_padded"])
    exact_match = __df_column_to_numpy(df["exact_match_padded"])
    id_x = __df_column_to_numpy(df["id"])
    label = None
    if "label_padded" in df:
        label = __df_column_to_numpy(df["label_padded"])

    evaluation_id_x = id_x
    evaluation_passage = __df_column_to_numpy(df["word_tokens_passage"])
    evaluation_question = __df_column_to_numpy(df["word_tokens_question"])

    tf = __cast_to_numpy_float(tf)
    pos = __cast_to_numpy_float(pos)
    ner = __cast_to_numpy_float(ner)
    passage = __cast_to_numpy_float(passage)
    question = __cast_to_numpy_float(question)
    exact_match = __cast_to_numpy_float(exact_match)
    if "label_padded" in df:
        label = __cast_to_numpy_float(label)

    return id_x, passage, question, pos, ner, tf, exact_match, label, evaluation_id_x, evaluation_passage, evaluation_question


def __export_df(df, onehot_pos, onehot_ner, glove_dim):
    cols = [
        "title", "word_tokens_passage_padded", "word_tokens_question_padded",
        "pos_padded", "ner_padded"
    ]
    if "answer" in df:
        cols.append("answer")
    df.drop(cols, inplace=True, axis=1)
    df = df.reset_index(drop=True)

    save_processed_data(df, onehot_pos, onehot_ner, glove_dim)


def get_data(glove_dim, ret="all", debug=False, json_path=None):
    create_tmp_directories()

    glove_matrix = load_glove_matrix(glove_dim)
    print("[Glove] downloaded.")

    wti = load_WTI(glove_dim)
    print("[WTI] prepared.")

    conf = load_config_data()

    df = None
    if debug is False and conf.get_argv_json_complete_name() is None:
        df = load_processed_data(wti, glove_dim)
    elif not conf.argv_changed(json_path, debug):
        df = load_processed_data(wti, glove_dim)
    else:
        clean_all_data_cache()

    # evaluation_data = load_evaluation_data_df()

    if glove_matrix is None or wti is None:
        glove = glove_reader(glove_dim)
        glove_matrix, wti = glove_embedding(glove, glove_dim)
        save_glove_matrix(glove_matrix, glove_dim)
        save_WTI(wti, glove_dim)

    if df is None:
        df = data_reader(json_path)
        print("[Data] downloaded.")

        # if evaluation_data is None:
        #     print("[DATA BACKUP] saving")
        #     save_evaluation_data_df(df)
        #     print("[DATA BACKUP] saved")

        if debug:
            df = df[0:5].copy()

        df, _ = reduce_mem_usage(df)
        df = data_preprocessing(df, wti)
        df, _ = reduce_mem_usage(df)
        df, onehot_pos, onehot_ner = add_features(df, wti)
        df, _ = reduce_mem_usage(df)
        print("[Data] processed.")

        __export_df(df, onehot_pos, onehot_ner, glove_dim)
        # conf.set_argv_json_complete_name(json_path, debug)
        # save_config_data(conf)
        print("[Data] exported.")
    else:
        print("[Data] loaded.")

    conf.set_argv_json_complete_name(json_path, debug)
    save_config_data(conf)
    # evaluation_data = load_evaluation_data_df()
    # if evaluation_data is None:
    #     print("[DATA BACKUP] saving")
    #     evaluation_data = save_evaluation_data_df(data_reader(json_path))
    #     print("[DATA BACKUP] saved")

    df_np = __data_to_numpy(df)

    if ret == "original":
        df_np = df_np[8], df_np[9], df_np[10]
    elif ret == "label":
        df_np = df_np[7]
    elif ret == "data":
        df_np = df_np[0], df_np[1], df_np[2], df_np[3], df_np[4], df_np[5], df_np[6]
    else:
        pass

    return df, df_np, glove_matrix, wti
