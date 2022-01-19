import os
import data_utils

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


def download_glove():
    DRIVE_ID = "15mTrPUQ4PAxfepzmRZfXNeKOJ3AubXrJ"
    RAW_FILE = os.path.join(TRAINING_DATA_LOCAL_DIR, "GloVe.txt")
    REQUIRED_FILE = os.path.join(TMP_TRAIN_DATA_DIR, "GloVe.6B.50d.txt")
    ZIP_FILE = os.path.join(TMP_TRAIN_DATA_DIR, "GloVe.6B.zip")

    create_tmp_directories()
    download_data(DRIVE_ID, ZIP_FILE, REQUIRED_FILE)
    data_utils.copy_data(REQUIRED_FILE, RAW_FILE)


download_training_set()
download_glove()