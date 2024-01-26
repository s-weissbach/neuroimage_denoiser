import shutil
import os


def ignore_files(dir: str, files: list[str]):
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]


def copy_folder_structure(directory_from: str, directory_to: str) -> None:
    shutil.copytree(
        directory_from, directory_to, ignore=ignore_files, dirs_exist_ok=True
    )
