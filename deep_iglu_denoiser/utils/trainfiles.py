import os
import pandas as pd
import numpy as np
import h5py
from alive_progress import alive_bar
import utils.normalization as normalization
from utils.activitymap import get_frames_position
from utils.open_file import open_file


class TrainFiles:
    def __init__(self, train_csv_path: str, overwrite: bool = False) -> None:
        """
        Initialize TrainFiles object.

        Args:
            train_csv_path (str): Path to the CSV file containing training examples information.
            overwrite (bool, optional): If True, overwrite existing files. Default is False.
        """
        self.train_csv_path = train_csv_path
        self.overwrite = overwrite
        self.file_list = {}
        if os.path.exists(self.train_csv_path):
            self.open_csv()
        else:
            self.train_examples = pd.DataFrame(
                columns=[
                    "h5_idx",
                    "original_filepath",
                    "target_frame",
                    "y_pos",
                    "x_pos",
                ]
            )

    def open_csv(self) -> None:
        """
        Open the CSV file containing training examples information.
        """
        if os.path.exists(self.train_csv_path):
            self.train_examples = pd.read_csv(self.train_csv_path)
        else:
            print(f"CSV path not found: {self.train_csv_path}")

    def files_to_traindata(
        self,
        directory: str,
        fileendings: list[str],
        min_z_score: float,
        before: int,
        after: int,
        crop_size: int,
        roi_size: int,
        output_h5_file: str,
        normalization_window_size: int = 50,
        foreground_background_split: float = 0.1,
    ):
        """
        Iterates through given directory and searches for all files having the specified fileendings.
        Opens these images and extracts active/inactive patches (size: crop_size) from the files
        (above/below min_z_score) and writes a single h5 file for training. active: foreground; inactive:
        background are splitted according to foreground_background_split. A train example consists of n_pre
        and n_post frames around the target frame.

        Args:
            directory (str): Directory containing image files.
            fileendings (list[str]): List of file endings to consider.
            min_z_score (float): Minimum Z-score threshold.
            before (int): Number of frames before the target frame.
            after (int): Number of frames after the target frame.
            crop_size (int): Size of the crop used for training.
            roi_size (int): Size of the region of interest.
            output_h5_file (str): Path to the output H5 file.
            window_size (int, optional): Size of the rolling window for normalization. Default is 50.
            foreground_background_split (float, optional): Split ratio for foreground and background patches.
                Default is 0.1.
        """
        train_example_list = []
        files_to_do = []
        for root, _, files in os.walk(directory):
            for file in files:
                if not any([file.endswith(ending) for ending in fileendings]):
                    continue
                files_to_do.append(os.path.join(root, file))
        if os.path.exists(output_h5_file) and not self.overwrite:
            print("Found existing h5-file. Will append.")
            hf = h5py.File(output_h5_file, "w")
            idx = np.max([int(key) for key in hf.keys()]) + 1
        elif os.path.exists(output_h5_file) and self.overwrite:
            os.remove(output_h5_file)
            hf = h5py.File(output_h5_file, "w")
            idx = 0
        else:
            hf = h5py.File(output_h5_file, "w")
            idx = 0
        print(f"Found {len(files_to_do)} file(s).")
        import time

        with alive_bar(len(files_to_do)) as bar:
            for filepath in files_to_do:
                if (
                    self.train_examples[
                        self.train_examples["original_filepath"] == filepath
                    ].shape[0]
                    > 0
                ):
                    print(f"Skipepd {filepath} - already in h5 file.")
                    bar()
                    continue
                try:
                    start = time.time()
                    tmp_file = open_file(filepath)
                    print(f"Opening file: {round(time.time()-start,4)}s")
                    start = time.time()
                    # find train examples with activity
                    tmp_file_rolling_normalization = (
                        normalization.rolling_window_z_norm(
                            tmp_file, normalization_window_size
                        )
                    )
                    print(
                        f"Rolling window normalization: {round(time.time()-start,4)}s"
                    )
                    start = time.time()
                    # will go through all frames and extract events that within a meaned kernel exceed the
                    # min_z_score threshold
                    # returns a list of events in the form [frame, y-coord, x-coord]
                    frames_and_positions = get_frames_position(
                        tmp_file_rolling_normalization,
                        min_z_score,
                        before,
                        after,
                        crop_size,
                        roi_size,
                        foreground_background_split,
                    )
                    print(f"Frames and positons: {round(time.time()-start,4)}s")
                    print(
                        f"Found {len(frames_and_positions)} example(s) in file {filepath}"
                    )
                    if len(frames_and_positions) == 0:
                        continue
                    start = time.time()
                    mean = np.mean(tmp_file, axis=0)
                    std = np.std(tmp_file, axis=0)
                    print(f"Mean and std: {round(time.time()-start,4)}s")
                    start = time.time()
                    tmp_file = normalization.z_norm(tmp_file, mean, std)
                    print(f"Normalization: {round(time.time()-start,4)}s")
                    start = time.time()
                    # create dict to be stored as h5 file
                    for event in frames_and_positions:
                        target_frame, y_pos, x_pos = event
                        train_example_list.append(
                            [str(idx), filepath, target_frame, y_pos, x_pos]
                        )
                        example = tmp_file[
                            target_frame,
                            y_pos : y_pos + crop_size,
                            x_pos : x_pos + crop_size,
                        ]
                        hf.create_dataset(str(idx), data=example)
                        idx += 1
                    print(f"Write to h5 file: {round(time.time()-start,4)}s")
                    bar()
                except:
                    print(f"Skipped file {filepath}")
                    bar()
            start = time.time()
        hf.close()
        self.train_examples = pd.DataFrame(
            train_example_list,
            columns=["h5_idx", "original_filepath", "target_frame", "y_pos", "x_pos"],
        )
        self.train_examples.to_csv(self.train_csv_path)
