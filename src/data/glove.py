import os
import numpy as np

from features import WordToIndex

from utils.data import copy_data, download_data, get_data_dir, get_tmp_data_dir
from utils.data_storage import create_tmp_directories

###


class GloVe:

    # @staticmethod
    # def __OOV_embedding(token):
    #     return np.array([0] * 50)

    #

    @staticmethod
    def download(glove_dim: int):
        DRIVE_ID = "15mTrPUQ4PAxfepzmRZfXNeKOJ3AubXrJ"
        RAW_FILE = os.path.join(get_data_dir(), "GloVe_" + str(glove_dim) + ".txt")
        REQUIRED_FILE = os.path.join(get_tmp_data_dir(), "glove.6B." + str(glove_dim) + "d.txt")
        ZIP_FILE = os.path.join(get_tmp_data_dir(), "GloVe.6B.zip")

        create_tmp_directories()
        download_data(DRIVE_ID, ZIP_FILE, REQUIRED_FILE)
        copy_data(REQUIRED_FILE, RAW_FILE)
        return RAW_FILE

    @staticmethod
    def load(glove_local_file):
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

    @staticmethod
    def embeddings_matrix(glove_embeddings, glove_dim):
        PAD_WORD = WordToIndex.PAD_WORD
        wti = WordToIndex()
        wti.fit_on_list(glove_embeddings.keys())

        TOKENIZER_MAX_WORD_INDEX = wti.get_index_len()
        embeddings_matrix = np.zeros((TOKENIZER_MAX_WORD_INDEX, glove_dim))

        for token in glove_embeddings.keys():
            word_index = wti.get_word_index(token)
            embeddings_matrix[word_index] = glove_embeddings[token]

        embeddings_matrix[0] = glove_embeddings[PAD_WORD]
        return embeddings_matrix, wti

    # @staticmethod
    # def inject_OOV_embeddings(df: pd.DataFrame, dictionary):
    #     emb_dict = copy(dictionary)
    #     claims_tokens = np.hstack(df['Claim'].to_list())
    #     evidence_tokens = np.hstack(df['Evidence'].to_list())
    #     tokens = np.concatenate((claims_tokens, evidence_tokens))
    #     for token in np.unique(tokens):
    #         if token not in emb_dict:
    #             emb_dict[token] = GloVe.__OOV_embedding(token)
    #     return emb_dict
