import os
import sys
import json
import pandas as pd

from utils.data import copy_data
from utils.data import create_tmp_directories, download_data
from utils.data import get_data_dir, get_tmp_data_dir
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


def load_training_set():
    print("Data downloading")
    raw_file = ""
    if len(sys.argv) <= 1:
        raw_file = download_training_set()
    else:
        raw_file = sys.argv[1]
    if not os.path.exists(raw_file):
        raise Exception(raw_file + " does not exists.")
    print("Data downloaded at position: " + raw_file + "\n")
    print("Converting json to dataframe")

    with open(raw_file, 'r', encoding="utf8", errors='ignore') as j:
        contents = json.loads(j.read().encode('utf-8').strip(), encoding='unicode_escape')

    contents = contents["data"]
    df = pd.json_normalize(
        contents, ['paragraphs', 'qas', 'answers'],
        ["title", ["paragraphs", "context"], ["paragraphs", "qas", "question"]]
    )
    df = df[["title", "paragraphs.context", "paragraphs.qas.question", "text", "answer_start"]]

    df.rename(
        columns={
            'title': 'title',
            'paragraphs.context': 'passage',
            'paragraphs.qas.question': 'question',
            'text': 'answer',
            'answer_start': 'answer_start'
        },
        inplace=True
    )
    print("Converted json to dataframe \n")
    return df


def data_reader():
    df = load_training_set()
    return df


def glove_reader(glove_dim):
    glove_file = download_glove(glove_dim)
    glove = load_glove(glove_file)
    return glove

