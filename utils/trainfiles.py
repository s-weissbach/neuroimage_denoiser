import os
import yaml
import numpy as np
from tqdm import tqdm
import utils.normalization as normalization
from utils.activitymap import get_frames_position
from utils.open_file import open_file


class TrainFiles:
    def __init__(self, train_yaml_path: str, overwrite: bool = False) -> None:
        self.train_yaml_path = train_yaml_path
        self.overwrite = overwrite
        self.file_dict = {}
        if os.path.exists(self.train_yaml_path):
            self.open_yaml()

    def open_yaml(self) -> None:
        if os.path.exists(self.train_yaml_path):
            with open(self.train_yaml_path, "r") as f:
                self.file_dict = yaml.safe_load(f)
        else:
            print(f"YAML path not found: {self.train_yaml_path}")

    def write_yaml(self) -> None:
        with open(self.train_yaml_path, "w") as f:
            yaml.dump(self.file_dict, f)

    def find_files(
        self,
        directory: str,
        fileendings: list[str],
        min_z_score: float,
        kernel_size: int,
        window_size: int = 50,
        n_threads: int = 1,
        foreground_background_split: float = 0.1,
    ):
        self.file_dict = {}
        files_to_do = []
        for root, _, files in os.walk(directory):
            for file in files:
                if not any([file.endswith(ending) for ending in fileendings]):
                    continue
                files_to_do.append(os.path.join(root, file))
        for idx, filepath in tqdm(enumerate(files_to_do), total=len(files_to_do)):
            tmp_file = open_file(filepath)
            mean = np.mean(tmp_file)
            std = np.std(tmp_file)
            # find train examples with activity
            tmp_file = normalization.rolling_window_z_norm(
                tmp_file, window_size, n_threads=n_threads
            )
            # will go through all frames and extract events that within a meaned kernel exceed the
            # min_z_score threshold
            # returns a list of events in the form [frame, y-coord, x-coord]
            frames_and_positions = get_frames_position(
                tmp_file, min_z_score, kernel_size, foreground_background_split
            )
            self.file_dict[idx] = {
                "filepath": filepath,
                "shape": list(tmp_file.shape),
                "mean": float(mean),
                "std": float(std),
                "frames_and_positions": frames_and_positions,
            }
        if self.overwrite:
            self.write_yaml()
