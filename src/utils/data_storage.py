import os
import pickle
import pandas as pd
import pickle as pkl
from data.dataframe_compression import DataframeCompression
from features.one_hot_encoder import OneHotEncoder
from data.config import Configuration
from utils.data import add_glove_dim_to_name, get_processed_data_dir, get_data_dir, get_tmp_data_dir

EVALUATION_DATA_FILE_NAME = "evaluation_data.pkl"
FINAL_DATA_FILE_NAME = "data.pkl"
GLOVE_MATRIX_FILE_NAME = "glove_matrix.pkl"
WORD_TO_INDEX_FILE_NAME = "word_to_index.pkl"
CONFIG_FILE_NAME = "configuration.pkl"


def create_tmp_directories():
    if not os.path.exists(get_tmp_data_dir()):
        os.mkdir(get_tmp_data_dir())

    if not os.path.exists(get_data_dir()):
        os.makedirs(get_data_dir())

    if not os.path.exists(get_processed_data_dir()):
        os.makedirs(get_processed_data_dir())


def clean_all_data_cache():
    folder = get_processed_data_dir()
    os.remove(os.path.join(folder, EVALUATION_DATA_FILE_NAME))
    for name in os.listdir(folder):
        if name.startswith("data"):
            os.remove(os.path.join(folder, name))


def save_processed_data(df: pd.DataFrame, OHE_pos: OneHotEncoder, OHE_ner: OneHotEncoder, glove_dim: str):
    name = add_glove_dim_to_name(FINAL_DATA_FILE_NAME, glove_dim)
    folder = get_processed_data_dir()
    file = os.path.join(folder, name)
    df_c = DataframeCompression(OHE_pos, OHE_ner)
    df_c.compress(df)
    with open(file, "wb") as handle:
        pickle.dump(df_c, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_processed_data(WTI, glove_dim: str):
    name = add_glove_dim_to_name(FINAL_DATA_FILE_NAME, glove_dim)
    folder = get_processed_data_dir()
    file = os.path.join(folder, name)
    if not os.path.exists(file):
        return None
    with open(file, "rb") as handle:
        return pickle.load(handle).extract(WTI)


def save_pickle(obj, file_name: str, folder: str):
    file = os.path.join(folder, file_name)

    with open(file, "wb") as f:
        pkl.dump(obj, f)


def load_pickle(file_name: str, folder: str):
    file = os.path.join(folder, file_name)
    if not os.path.exists(file):
        return None

    with open(file, "rb") as f:
        return pkl.load(f)


def save_evaluation_data_data(evaluation_data):
    save_pickle(evaluation_data, EVALUATION_DATA_FILE_NAME, get_processed_data_dir())


def load_evaluation_data_data():
    return load_pickle(EVALUATION_DATA_FILE_NAME, get_processed_data_dir())


def save_config_data(config):
    save_pickle(config, CONFIG_FILE_NAME, get_data_dir())


def load_config_data():
    if os.path.exists(os.path.join(get_data_dir(), CONFIG_FILE_NAME)):
        return load_pickle(CONFIG_FILE_NAME, get_data_dir())
    else:
        return Configuration()


def save_glove_matrix(glove_matrix, glove_dim: str):
    name = add_glove_dim_to_name(GLOVE_MATRIX_FILE_NAME, glove_dim)
    save_pickle(glove_matrix, name, get_processed_data_dir())


def load_glove_matrix(glove_dim: str):
    name = add_glove_dim_to_name(GLOVE_MATRIX_FILE_NAME, glove_dim)
    return load_pickle(name, get_processed_data_dir())


def save_WTI(WTI, glove_dim: str):
    name = add_glove_dim_to_name(WORD_TO_INDEX_FILE_NAME, glove_dim)
    save_pickle(WTI, name, get_processed_data_dir())


def load_WTI(glove_dim: str):
    name = add_glove_dim_to_name(WORD_TO_INDEX_FILE_NAME, glove_dim)
    return load_pickle(name, get_processed_data_dir())
