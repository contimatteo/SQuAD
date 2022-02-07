import os
import sys

from google_drive_downloader import GoogleDriveDownloader as gdd
from shutil import copyfile
import nltk

###


def nltk_download_utilities():
    # nltk.download('tagsets')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('omw-1.4')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    # spacy.cli.download("en_core_web_sm")


def get_project_directory():
    return cd_parent(cd_parent(cd_parent(os.path.realpath(__file__))))


def download_url(drive_id, save_path, unzip=True):
    gdd.download_file_from_google_drive(file_id=drive_id, dest_path=save_path, unzip=unzip)


def get_data_dir():
    return os.path.join(get_project_directory(), "data", "raw")


def get_processed_data_dir():
    return os.path.join(get_project_directory(), "data", "processed")


def get_tmp_data_dir():
    return os.path.join(get_project_directory(), "tmp")


def get_default_raw_file_name():
    return os.path.join(get_data_dir(), "training_set.json")


def cd_parent(file):
    return os.path.dirname(file)


def copy_data(from_file, to_file):
    if not os.path.exists(to_file):
        copyfile(from_file, to_file)


def download_data(drive_id, zip_file_name, required_file_name):
    if not os.path.exists(required_file_name):
        if os.path.exists(zip_file_name):
            os.remove(zip_file_name)
        download_url(drive_id, zip_file_name)


def add_glove_dim_to_name(file_name, glove_dim: str):
    return file_name.replace(".", f"_{glove_dim}.")


def get_argv():
    if len(sys.argv) > 1:
        return sys.argv[1]
    else:
        return None


