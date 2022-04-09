import os
import pickle
import pandas as pd

from data import DataframeCompression
from features.one_hot_encoder import OneHotEncoder
from utils import DataUtils
# from utils.data import get_data_dir

###

EVALUATION_DATA_FILE_NAME = "evaluation_data.pkl"
GLOVE_MATRIX_FILE_NAME = "glove_matrix.pkl"
WORD_TO_INDEX_FILE_NAME = "word_to_index.pkl"
CONFIG_FILE_NAME = "configuration.pkl"

###


def create_tmp_directories():
    if not os.path.exists(DataUtils.tmp_data_dir()):
        os.mkdir(DataUtils.tmp_data_dir())

    if not os.path.exists(DataUtils.data_dir()):
        os.makedirs(DataUtils.data_dir())

    if not os.path.exists(DataUtils.processed_data_dir()):
        os.makedirs(DataUtils.processed_data_dir())


def clean_all_data_cache():
    folder = DataUtils.processed_data_dir()
    if os.path.exists(os.path.join(folder, EVALUATION_DATA_FILE_NAME)):
        os.remove(os.path.join(folder, EVALUATION_DATA_FILE_NAME))
    for name in os.listdir(folder):
        if name.startswith("data"):
            os.remove(os.path.join(folder, name))


def save_processed_data(
    df: pd.DataFrame, ohe_pos: OneHotEncoder, ohe_ner: OneHotEncoder, glove_dim: str, file_name: str
):
    name = DataUtils.build_glove_file_name_by_dim(file_name, glove_dim)
    folder = DataUtils.processed_data_dir()
    file = os.path.join(folder, name)

    df_c = DataframeCompression(ohe_pos.get_ohe_dicts(), ohe_ner.get_ohe_dicts())
    df_c.compress(df)

    df_c = df_c.to_pickle()
    with open(file, "wb") as handle:
        pickle.dump(df_c, handle)


def load_processed_data(glove_dim, file_name: str):
    name = DataUtils.build_glove_file_name_by_dim(file_name, glove_dim)
    folder = DataUtils.processed_data_dir()
    file = os.path.join(folder, name)
    if not os.path.exists(file):
        return None
    with open(file, "rb") as handle:
        df_c = DataframeCompression()
        df_c.from_pickle(pickle.load(handle))
        return df_c.extract()


def save_pickle(obj, file_name: str, folder: str):
    file = os.path.join(folder, file_name)

    with open(file, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(file_name: str, folder: str):
    file = os.path.join(folder, file_name)
    if not os.path.exists(file):
        return None

    with open(file, "rb") as f:
        return pickle.load(f)


def save_evaluation_data_data(evaluation_data):
    save_pickle(evaluation_data, EVALUATION_DATA_FILE_NAME, DataUtils.processed_data_dir())


def load_evaluation_data_data():
    return load_pickle(EVALUATION_DATA_FILE_NAME, DataUtils.processed_data_dir())


def save_config_data(config):
    save_pickle(config, CONFIG_FILE_NAME, DataUtils.data_dir())


# def load_config_data():
#     if os.path.exists(os.path.join(DataUtils.data_dir(), CONFIG_FILE_NAME)):
#         return load_pickle(CONFIG_FILE_NAME, DataUtils.data_dir())
#     else:
#         return Configuration()


def save_glove_matrix(glove_matrix, glove_dim):
    name = DataUtils.build_glove_file_name_by_dim(GLOVE_MATRIX_FILE_NAME, glove_dim)

    save_pickle(glove_matrix, name, DataUtils.processed_data_dir())


def load_glove_matrix(glove_dim):
    name = DataUtils.build_glove_file_name_by_dim(GLOVE_MATRIX_FILE_NAME, glove_dim)

    return load_pickle(name, DataUtils.processed_data_dir())


def save_wti(wti, glove_dim):
    name = DataUtils.build_glove_file_name_by_dim(WORD_TO_INDEX_FILE_NAME, glove_dim)

    save_pickle(wti, name, DataUtils.processed_data_dir())


def load_wti(glove_dim):
    name = DataUtils.build_glove_file_name_by_dim(WORD_TO_INDEX_FILE_NAME, glove_dim)

    return load_pickle(name, DataUtils.processed_data_dir())
