import os

import numpy as np
import pandas as pd
from google_drive_downloader import GoogleDriveDownloader as gdd
from shutil import copyfile
import nltk
import pickle as pkl


def nltk_download_utilities():
    # nltk.download('tagsets')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('omw-1.4')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    # spacy.cli.download("en_core_web_sm")


FINAL_DATA_FILE_NAME = "data.csv"
GLOVE_MATRIX_FILE_NAME = "glove_matrix.pkl"
WORD_TO_INDEX_FILE_NAME = "word_to_index.pkl"


def get_project_directory():
    return cd_parent(cd_parent(cd_parent(os.path.realpath(__file__))))


def download_url(drive_id, save_path):
    gdd.download_file_from_google_drive(file_id=drive_id,
                                        dest_path=save_path,
                                        unzip=True)


def get_data_dir():
    return os.path.join(get_project_directory(), "data", "raw")


def get_processed_data_dir():
    return os.path.join(get_project_directory(), "data", "processed")


def get_tmp_data_dir():
    return os.path.join(get_project_directory(), "tmp")


def cd_parent(file):
    return os.path.dirname(file)


def copy_data(from_file, to_file):
    if not os.path.exists(to_file):
        copyfile(from_file, to_file)


def create_tmp_directories():
    if not os.path.exists(get_tmp_data_dir()):
        os.mkdir(get_tmp_data_dir())

    if not os.path.exists(get_data_dir()):
        os.makedirs(get_data_dir())

    if not os.path.exists(get_processed_data_dir()):
        os.makedirs(get_processed_data_dir())


def download_data(drive_id, zip_file_name, required_file_name):
    if not os.path.exists(required_file_name):
        if os.path.exists(zip_file_name):
            os.remove(zip_file_name)
        download_url(drive_id, zip_file_name)


# def save_processed_data(df: pd.DataFrame, glove_dim: str):
#     name = add_glove_dim_to_name(FINAL_DATA_FILE_NAME, glove_dim)
#     folder = get_processed_data_dir()
#     df.to_pickle(os.path.join(folder, name))
#
#
# def load_processed_data(glove_dim: str):
#     name = add_glove_dim_to_name(FINAL_DATA_FILE_NAME, glove_dim)
#     folder = get_processed_data_dir()
#     file = os.path.join(folder, name)
#     if not os.path.exists(file):
#         return None
#
#     return pd.read_pickle(file)


def dataframe_array_to_string(df: pd.DataFrame):
    COL_TO_EXCLUDE = ["title", "answer", "passage_index", "question_index", "chunk_index"]
    for col in df.columns:
        if col not in COL_TO_EXCLUDE:
            df[col] = df[col].apply(lambda a: np.array2string(a))
    return df


def save_processed_data(df: pd.DataFrame, glove_dim: str):
    # df = dataframe_array_to_string(df)
    name = add_glove_dim_to_name(FINAL_DATA_FILE_NAME, glove_dim)
    folder = get_processed_data_dir()
    df.to_csv(os.path.join(folder, name))


def load_processed_data(glove_dim: str):
    name = add_glove_dim_to_name(FINAL_DATA_FILE_NAME, glove_dim)
    folder = get_processed_data_dir()
    file = os.path.join(folder, name)
    if not os.path.exists(file):
        return None

    return pd.read_csv(file)


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


def add_glove_dim_to_name(file_name, glove_dim: str):
    return file_name.replace(".", f"_{glove_dim}.")

