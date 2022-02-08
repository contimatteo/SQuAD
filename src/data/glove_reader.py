import os
import pandas as pd
import numpy as np

from copy import copy
from features.word_to_index import WordToIndex

from utils.data import copy_data, download_data, get_data_dir, get_tmp_data_dir
from utils.data_storage import create_tmp_directories

###


def download_glove(glove_dim: int):
    DRIVE_ID = "15mTrPUQ4PAxfepzmRZfXNeKOJ3AubXrJ"
    RAW_FILE = os.path.join(get_data_dir(), "GloVe_" + str(glove_dim) + ".txt")
    REQUIRED_FILE = os.path.join(get_tmp_data_dir(), "glove.6B." + str(glove_dim) + "d.txt")
    ZIP_FILE = os.path.join(get_tmp_data_dir(), "GloVe.6B.zip")

    create_tmp_directories()
    download_data(DRIVE_ID, ZIP_FILE, REQUIRED_FILE)
    copy_data(REQUIRED_FILE, RAW_FILE)
    return RAW_FILE


def load_glove(glove_local_file):
    glove_embeddings = {}
    print("Loading Glove Model")
    with open(glove_local_file, encoding="utf8") as f:
        lines = f.readlines()
    print("Loaded Glove Model")
    print("Parsing Glove Model")
    for line in lines:
        splits = line.split()
        word = splits[0]
        embedding = np.array([float(val) for val in splits[1:]])
        glove_embeddings[word] = embedding
    print("Parsing Done.\n")

    print(len(glove_embeddings.keys()), "words loaded from GloVe.")
    return glove_embeddings


def compute_OOV_token_embedding(token):
    return np.array([0] * 50)


def inject_OOV_embeddings(df: pd.DataFrame, dictionary):
    emb_dict = copy(dictionary)

    claims_tokens = np.hstack(df['Claim'].to_list())
    evidence_tokens = np.hstack(df['Evidence'].to_list())
    tokens = np.concatenate((claims_tokens, evidence_tokens))

    for token in np.unique(tokens):
        if token not in emb_dict:
            emb_dict[token] = compute_OOV_token_embedding(token)

    return emb_dict


def glove_embedding(glove_embeddings, glove_dim):
    WTI = WordToIndex()
    WTI.fit_on_list(glove_embeddings.keys())

    TOKENIZER_MAX_WORD_INDEX = WTI.get_index_len(
    )  # np.array(list(tokenizer.word_index.values())).max()
    embeddings_matrix = np.zeros((TOKENIZER_MAX_WORD_INDEX, glove_dim))

    for token in glove_embeddings.keys():
        word_index = WTI.get_word_index(token)
        embeddings_matrix[word_index] = glove_embeddings[token]

    return embeddings_matrix, WTI
