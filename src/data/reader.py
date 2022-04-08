import os
import json
import pandas as pd

from utils.data import copy_data, download_data, get_tmp_data_dir
# from utils.data_storage import save_evaluation_data_data, load_evaluation_data_data
from utils.data import get_default_raw_file_name

from .glove import GloVe

###


class DataReader:

    @staticmethod
    def __download_training_set():
        DRIVE_ID = "19byT_6Hhx4Di1pzbd6bmxQ8sKwCSPhqg"
        RAW_FILE = get_default_raw_file_name()
        REQUIRED_FILE = os.path.join(get_tmp_data_dir(), "training_set.json")
        ZIP_FILE = os.path.join(get_tmp_data_dir(), "training_set.zip")

        download_data(DRIVE_ID, ZIP_FILE, REQUIRED_FILE)
        copy_data(REQUIRED_FILE, RAW_FILE)
        return RAW_FILE

    @staticmethod
    def __load_dataset(json_path):
        print("Data downloading")
        raw_file = ""
        if json_path is None:
            raw_file = DataReader.__download_training_set()
        else:
            raw_file = json_path
        if not os.path.exists(raw_file):
            raise Exception(raw_file + " does not exists.")
        print("Data downloaded at position: " + raw_file + "\n")
        print("Converting json to dataframe")

        with open(raw_file, 'r', encoding="utf-8", errors='ignore') as j:
            contents = json.loads(j.read().encode('utf-8').strip(), encoding='unicode_escape')

        contents = contents["data"]
        df = pd.json_normalize(
            contents, ['paragraphs', 'qas'], ["title", ["paragraphs", "context"]]
        )
        if "answers" in df:
            df = df[["id", "title", "paragraphs.context", "question", "answers"]]
            # df["answers"] = [i[0] for i in df["answers"]]
            A = df['answers'].apply(lambda x: pd.Series(x[0])).add_prefix('answers.')
            df = df.join([A])
            df.drop(columns='answers', inplace=True)
            df.rename(
                columns={
                    'id': 'id',
                    'title': 'title',
                    'paragraphs.context': 'passage',
                    'question': 'question',
                    'answers.text': 'answer',
                    'answers.answer_start': 'answer_start'
                },
                inplace=True
            )
        else:
            df = df[["id", "title", "paragraphs.context", "question"]]

            df.rename(
                columns={
                    'id': 'id',
                    'title': 'title',
                    'paragraphs.context': 'passage',
                    'question': 'question'
                },
                inplace=True
            )

        print("Converted json to dataframe \n")
        return df

    #

    @staticmethod
    def dataset(json_path) -> pd.DataFrame:
        return DataReader.__load_dataset(json_path)

    @staticmethod
    def glove(glove_dim):
        glove_file = GloVe.download(glove_dim)
        glove = Glove.load(glove_file)
        return glove

    # @staticmethod

    # def only_dict(d):
    #     import ast
    #     return ast.literal_eval(str(d))

    # @staticmethod
    # def save_evaluation_data_df(df: pd.DataFrame):
    #     evaluation_df = df[["id", "passage"]].copy()
    #     save_evaluation_data_data(df[["id", "passage"]].copy())
    #     return evaluation_df

    # @staticmethod
    # def load_evaluation_data_df():
    #     return load_evaluation_data_data()