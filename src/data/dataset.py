import os.path
import pandas as pd

import utils.configs as Configs

from features import Features
from utils.data_storage import save_processed_data, load_processed_data
from utils.data_storage import save_glove_matrix, load_glove_matrix
from utils.data_storage import save_wti, load_wti, create_tmp_directories
from utils import MemoryUtils

from .preprocessing import DataPreprocessing
from .reader import DataReader
from .glove import GloVe

###

df_np = None
glove_matrix = None

###


class Dataset:

    @staticmethod
    def __delete_cache():
        from .cleaner import delete_cache_cleaner
        from features import Lemmatizer, NER, POS, TermFrequency

        delete_cache_cleaner()

        Lemmatizer.delete_cache()
        NER.delete_cache()
        POS.delete_cache()
        TermFrequency.delete_cache()

    @staticmethod
    def __data_to_list(df: pd.DataFrame):
        tf = df["term_frequency_padded"]
        pos = df["pos_onehot_padded"]
        ner = df["ner_onehot_padded"]
        passage = df["word_index_passage_padded"]
        question = df["word_index_question_padded"]
        exact_match = df["exact_match_padded"]
        question_id = df["id"]
        passage_id = df["passage_index"]
        label = df["label_padded"] if "label_padded" in df else None
        mask_p = df["mask_passage_padded"]
        mask_q = df["mask_question_padded"]

        evaluation_passage = df["word_tokens_passage_with_spaces"]
        evaluation_question = df["word_tokens_question"]

        return question_id, passage, question, pos, ner, tf, exact_match, label, question_id, evaluation_passage, evaluation_question, passage_id, mask_p, mask_q

    @staticmethod
    def __export_df(df, onehot_pos, onehot_ner, glove_dim, file_name):
        cols = [
            "title", "word_tokens_passage_padded", "word_tokens_question_padded", "pos_padded",
            "ner_padded"
        ]
        if "answer" in df:
            cols.append("answer")
        df.drop(cols, inplace=True, axis=1)
        df = df.reset_index(drop=True)

        save_processed_data(df, onehot_pos, onehot_ner, glove_dim, file_name=file_name)

    #

    @staticmethod
    def optimize_memory():
        global df_np, glove_matrix

        df_np = None
        glove_matrix = None

    @staticmethod
    def load(debug=False, json_path=None):
        global df_np, glove_matrix

        create_tmp_directories()
        glove_dim = Configs.DIM_EMBEDDING

        glove_matrix = load_glove_matrix(glove_dim)
        print("[Glove] downloaded.")

        wti = load_wti(glove_dim)
        print("[WTI] prepared.")

        file_name = "drive_dataset.pkl"
        if json_path is not None:
            file_name = os.path.basename(json_path).replace(".json", ".pkl")
        df = load_processed_data(glove_dim, file_name=file_name)

        if glove_matrix is None or wti is None:
            glove = DataReader.glove(glove_dim)
            glove_matrix, wti = GloVe.embeddings_matrix(glove, glove_dim)
            save_glove_matrix(glove_matrix, glove_dim)
            save_wti(wti, glove_dim)

        if df is None:
            df = DataReader.dataset(json_path)
            print("[Data] downloaded.")

            if debug:
                df = df[0:100].copy()

            df, _ = MemoryUtils.reduce_df_storage(df)
            df = DataPreprocessing.apply(df, wti)
            df, _ = MemoryUtils.reduce_df_storage(df)
            df, onehot_pos, onehot_ner = Features.build(df, wti)
            df, _ = MemoryUtils.reduce_df_storage(df)
            print("[Data] processed.")

            Dataset.__export_df(df, onehot_pos, onehot_ner, glove_dim, file_name)
            print("[Data] exported.")
        else:
            print("[Data] loaded.")

        print("Deleting cache")
        Dataset.__delete_cache()
        print("Deleted cache")

        print("[Data] converting to list")
        df_np = Dataset.__data_to_list(df)
        print("[Data] converted to list")

    @staticmethod
    def extract(ret: str):
        global df_np, glove_matrix

        assert df_np is not None
        assert glove_matrix is not None
        assert isinstance(ret, str)

        if ret == "original":
            return [df_np[8], df_np[9], df_np[10], df_np[11]]
        elif ret == "labels":
            return [df_np[7]]
        elif ret == "features":
            return [
                df_np[0], df_np[1], df_np[2], df_np[3], df_np[4], df_np[5], df_np[6], df_np[12],
                df_np[13]
            ]
        elif ret == "glove":
            return glove_matrix

        raise Exception("`get_data` invalid enum value.")
