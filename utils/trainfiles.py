import os
import tifffile
import json


class trainfiles:
    def __init__(self, train_json_path: str, overwrite: bool = False) -> None:
        self.train_json_path = train_json_path
        self.overwrite = overwrite
        self.file_dict = {}
        if os.path.exists(self.train_json_path):
            self.open_json()

    def open_json(self) -> None:
        if os.path.exists(self.train_json_path):
            with open(self.train_json_path, "r") as f:
                self.file_dict = json.load(f)
        else:
            print(f"JSON path not found: {self.train_json_path}")

    def write_json(self) -> None:
        with open(self.train_json_path, "w") as f:
            json.dump(self.file_dict, f)

    def find_files(self, directory: str, fileendings: list[str]):
        self.file_dict = {}
        idx = 0
        for root, _, files in os.walk(directory):
            for file in files:
                if not any([file.endswith(ending) for ending in fileendings]):
                    continue
                filepath = os.path.join(root, file)
                tmp_file = tifffile.imread(filepath)
                self.file_dict[idx] = {"filepath": filepath, "shape": tmp_file.shape}
                idx += 1
        if self.overwrite:
            self.write_json()
