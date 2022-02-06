import os
import sys
import json
import pandas as pd

from utils.data import copy_data
from utils.data import create_tmp_directories, download_data
from utils.data import get_data_dir, get_tmp_data_dir
from utils.data import save_og_data, load_og_data
from .glove_reader import load_glove, download_glove

###


def download_training_set():
    DRIVE_ID = "19byT_6Hhx4Di1pzbd6bmxQ8sKwCSPhqg"
    RAW_FILE = os.path.join(get_data_dir(), "training_set.json")
    REQUIRED_FILE = os.path.join(get_tmp_data_dir(), "training_set.json")
    ZIP_FILE = os.path.join(get_tmp_data_dir(), "training_set.zip")

    create_tmp_directories()
    download_data(DRIVE_ID, ZIP_FILE, REQUIRED_FILE)
    copy_data(REQUIRED_FILE, RAW_FILE)
    return RAW_FILE


def load_training_set(kwargs):
    print("Data downloading")
    raw_file = ""
    print(kwargs)
    if kwargs is None:
        raw_file = download_training_set()
    else:
        raw_file = kwargs[1]
    if not os.path.exists(raw_file):
        raise Exception(raw_file + " does not exists.")
    print("Data downloaded at position: " + raw_file + "\n")
    print("Converting json to dataframe")

    with open(raw_file, 'r', encoding="utf8", errors='ignore') as j:
        contents = json.loads(j.read().encode('utf-8').strip(), encoding='unicode_escape')

    contents = contents["data"]
    df = pd.json_normalize(
        contents, ['paragraphs', 'qas', 'answers'],
        ["title", ["paragraphs", "context"], ["paragraphs", "qas", "question"], ["paragraphs", "qas", "id"]]
    )
    df = df[["paragraphs.qas.id", "title", "paragraphs.context", "paragraphs.qas.question", "text", "answer_start"]]

    df.rename(
        columns={
            'title': 'title',
            'paragraphs.context': 'passage',
            'paragraphs.qas.question': 'question',
            'text': 'answer',
            'answer_start': 'answer_start',
            'paragraphs.qas.id': 'id'
        },
        inplace=True
    )
    print("Converted json to dataframe \n")
    return df


def save_og_df(df: pd.DataFrame):
    save_og_data(df[["id", "passage"]].copy())
    # print(load_og_data().head())


def load_og_df():
    return load_og_data()


def data_reader(kwargs):
    df = load_training_set(kwargs)
    return df


def glove_reader(glove_dim):
    glove_file = download_glove(glove_dim)
    glove = load_glove(glove_file)
    return glove

