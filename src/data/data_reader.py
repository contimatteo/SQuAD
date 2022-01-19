import os
import sys
import data_utils

import json
import numpy as np
import tensorflow as tf
import pandas as pd
# if glove_embeddings is None:
#   if not os.path.exists(GLOVE_LOCAL_DIR):
#     os.makedirs(GLOVE_LOCAL_DIR)
#
#   if not os.path.exists(GLOVE_LOCAL_FILE_ZIP):
#     urllib.request.urlretrieve(GLOVE_REMOTE_URL, GLOVE_LOCAL_FILE_ZIP)
#     print("Successful download")
#     tmp = Path(GLOVE_LOCAL_FILE_ZIP)
#     tmp.rename(tmp.with_suffix(".zip"))
#
#   with zipfile.ZipFile(f"{GLOVE_LOCAL_FILE_ZIP}.zip", 'r') as zip_ref:
#     zip_ref.extractall(path=GLOVE_LOCAL_DIR)
#     print("Successful extraction")


TRAINING_DATA_LOCAL_DIR = os.path.join(data_utils.get_project_directory(), "data", "raw")
TMP_TRAIN_DATA_DIR = os.path.join(data_utils.get_project_directory(), "tmp")


def create_tmp_directories():
    if not os.path.exists(TMP_TRAIN_DATA_DIR):
        os.mkdir(TMP_TRAIN_DATA_DIR)

    if not os.path.exists(TRAINING_DATA_LOCAL_DIR):
        os.makedirs(TRAINING_DATA_LOCAL_DIR)


def download_data(drive_id, zip_file_name, required_file_name):
    if not os.path.exists(required_file_name):
        if os.path.exists(zip_file_name):
            os.remove(zip_file_name)
        data_utils.download_url(drive_id, zip_file_name)
    print("Successful download")


def download_training_set():
    DRIVE_ID = "19byT_6Hhx4Di1pzbd6bmxQ8sKwCSPhqg"
    RAW_FILE = os.path.join(TRAINING_DATA_LOCAL_DIR, "training_set.json")
    REQUIRED_FILE = os.path.join(TMP_TRAIN_DATA_DIR, "training_set.json")
    ZIP_FILE = os.path.join(TMP_TRAIN_DATA_DIR, "training_set.zip")

    create_tmp_directories()
    download_data(DRIVE_ID, ZIP_FILE, REQUIRED_FILE)
    data_utils.copy_data(REQUIRED_FILE, RAW_FILE)
    return RAW_FILE


def download_glove():
    DRIVE_ID = "15mTrPUQ4PAxfepzmRZfXNeKOJ3AubXrJ"
    RAW_FILE = os.path.join(TRAINING_DATA_LOCAL_DIR, "GloVe.txt")
    REQUIRED_FILE = os.path.join(TMP_TRAIN_DATA_DIR, "GloVe.6B.50d.txt")
    ZIP_FILE = os.path.join(TMP_TRAIN_DATA_DIR, "GloVe.6B.zip")

    create_tmp_directories()
    download_data(DRIVE_ID, ZIP_FILE, REQUIRED_FILE)
    data_utils.copy_data(REQUIRED_FILE, RAW_FILE)
    return RAW_FILE

    
def load_training_set():
    raw_file = ""
    if len(sys.argv)<=1:
        raw_file = download_training_set()
    else:
        raw_file = sys.argv[1]
    
    print(raw_file)
    if not os.path.exists(raw_file):
        raise Exception(raw_file+" does not exists.")
        

  
    with open(raw_file, 'r') as j:
        contents = json.loads(j.read())

    #pc=['data','paragraphs','qas','answers']
    #js = pd.io.json.json_normalize(contents , pc )
    #m = pd.io.json.json_normalize(contents, pc[:-1] )
    #r = pd.io.json.json_normalize(contents,pc[:-2])

    #idx = np.repeat(r['context'].values, r.qas.str.len())
    #ndx  = np.repeat(m['id'].values,m['answers'].str.len())
    #m['context'] = idx
    #js['q_idx'] = ndx
    #main = pd.concat([ m[['id','question','context']].set_index('id'),js.set_index('q_idx')],1,sort=False).reset_index()
    #main['c_id'] = main['context'].factorize()[0]

    #print(main.head())
    #print(main.columns)



    contents = contents["data"]
    df = pd.json_normalize(contents,['paragraphs','qas','answers'],["title",["paragraphs","context"],["paragraphs","qas","question"]])
    df = df[["title","paragraphs.context","paragraphs.qas.question","text","answer_start"]]

    df.rename(columns = {'title':'title', 'paragraphs.context':'paragraph', 
                         'paragraphs.qas.question':'question', 'text':'answer', 'answer_start':'answer_start'}, inplace = True)
    
    return df
    
    
    
    
    
    
pd.set_option('display.max_columns', None)    
pd.set_option('display.max_colwidth', None)
df = load_training_set() 
print(df.columns)
print(df[0:1])
download_glove()