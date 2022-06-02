import os
import sys
import nltk

from google_drive_downloader import GoogleDriveDownloader as gdd
from shutil import copyfile
import argparse

###


class DataUtils:

    @staticmethod
    def __parent_dir(file):
        return os.path.dirname(file)

    @staticmethod
    def __root_dir():
        return DataUtils.__parent_dir(
            DataUtils.__parent_dir(DataUtils.__parent_dir(os.path.realpath(__file__)))
        )

    #

    @staticmethod
    def data_dir():
        return os.path.join(DataUtils.__root_dir(), "data", "raw")

    @staticmethod
    def processed_data_dir():
        return os.path.join(DataUtils.__root_dir(), "data", "processed")

    @staticmethod
    def tmp_data_dir():
        return os.path.join(DataUtils.__root_dir(), "tmp")

    @staticmethod
    def default_training_file_name():
        return os.path.join(DataUtils.data_dir(), "training_set.json")

    @staticmethod
    def copy_file_content_to(from_file, to_file):
        if not os.path.exists(to_file):
            copyfile(from_file, to_file)

    @staticmethod
    def build_glove_file_name_by_dim(file_name, glove_dim: str):
        return file_name.replace(".", f"_{glove_dim}.")

    @staticmethod
    def get_first_argv():
        if len(sys.argv) > 1:
            return sys.argv[1]
        else:
            return None

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(description="compute answers")
        parser.add_argument('test_json', type=str, help='name of json file to parse')
        return parser.parse_args()

    @staticmethod
    def get_input_file():
        args = DataUtils.parse_args()
        file_url = args.test_json

        assert os.path.exists(file_url)
        assert isinstance(file_url, str)
        assert len(file_url) > 5
        assert ".json" in file_url

        return file_url


###


class GoogleDriveUtils:

    @staticmethod
    def __download_file_by_id(drive_id, save_path, unzip=True):
        gdd.download_file_from_google_drive(file_id=drive_id, dest_path=save_path, unzip=unzip)

    @staticmethod
    def download_resource_by_id(drive_id, zip_file_name, required_file_name):
        if not os.path.exists(required_file_name):
            if os.path.exists(zip_file_name):
                os.remove(zip_file_name)

            GoogleDriveUtils.__download_file_by_id(drive_id, zip_file_name)


###


class NltkUtils:

    @staticmethod
    def download_utilities():
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        nltk.download('maxent_ne_chunker', quiet=True)
        nltk.download('words', quiet=True)
