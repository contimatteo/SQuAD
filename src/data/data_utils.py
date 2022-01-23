import os
from google_drive_downloader import GoogleDriveDownloader as gdd
from shutil import copyfile


def get_project_directory():
    return cd_parent(cd_parent(cd_parent(os.path.realpath(__file__))))


def download_url(drive_id, save_path):
    gdd.download_file_from_google_drive(file_id=drive_id,
                                        dest_path=save_path,
                                        unzip=True)


def cd_parent(file):
    return os.path.dirname(file)


def copy_data(from_file, to_file):
    if not os.path.exists(to_file):
        copyfile(from_file, to_file)
