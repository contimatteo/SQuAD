import os

import spacy.cli
from google_drive_downloader import GoogleDriveDownloader as gdd
from shutil import copyfile
import nltk


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


def download_url(drive_id, save_path):
    gdd.download_file_from_google_drive(file_id=drive_id,
                                        dest_path=save_path,
                                        unzip=True)


def get_data_dir():
    return os.path.join(get_project_directory(), "data", "raw")


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


def download_data(drive_id, zip_file_name, required_file_name):
    if not os.path.exists(required_file_name):
        if os.path.exists(zip_file_name):
            os.remove(zip_file_name)
        download_url(drive_id, zip_file_name)