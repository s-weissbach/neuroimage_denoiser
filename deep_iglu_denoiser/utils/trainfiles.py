import os
import pandas as pd
import numpy as np
import h5py
from alive_progress import alive_bar
import deep_iglu_denoiser.utils.normalization as normalization
from deep_iglu_denoiser.utils.activitymap import get_frames_position
from deep_iglu_denoiser.utils.open_file import open_file

import time


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
        window_size: int = 50,
        stimulationframes: list[int] = [],
        n_frames: int = 1,
        foreground_background_split: float = 0.1,
        memory_optimized=False,
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

        # find files recursively
        for root, _, files in os.walk(directory):
            for file in files:
                if not any([file.endswith(ending) for ending in fileendings]):
                    continue
                files_to_do.append(os.path.join(root, file))
        print(f"Found {len(files_to_do)} file(s).")

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

        with alive_bar(len(files_to_do)) as bar:
            for filepath in files_to_do:
                if (
                    self.train_examples[
                        self.train_examples["original_filepath"] == filepath
                    ].shape[0]
                    > 0
                ):
                    print(f"Skipped {filepath} - already in h5 file.")
                    bar()
                    continue

                if memory_optimized:
                    idx = self.handle_file_memory_optimized(
                        idx,
                        train_example_list,
                        filepath,
                        hf,
                        directory,
                        min_z_score,
                        before,
                        after,
                        crop_size,
                        roi_size,
                        output_h5_file,
                        window_size,
                        stimulationframes,
                        n_frames,
                        foreground_background_split,
                    )
                else:
                    idx = self.handle_file(
                        idx,
                        train_example_list,
                        filepath,
                        hf,
                        directory,
                        min_z_score,
                        before,
                        after,
                        crop_size,
                        roi_size,
                        output_h5_file,
                        window_size,
                        stimulationframes,
                        n_frames,
                        foreground_background_split,
                    )
                bar()

        if memory_optimized:
            os.remove(os.path.join(directory, "mmap_time_file.npy"))
            os.remove(os.path.join(directory, "mmap_time_znorm_file.npy"))
            os.remove(os.path.join(directory, "mmap_std.npy"))
            os.remove(os.path.join(directory, "mmap_mean.npy"))

        hf.close()

        self.train_examples = pd.DataFrame(
            train_example_list,
            columns=["h5_idx", "original_filepath", "target_frame", "y_pos", "x_pos"],
        )
        self.train_examples.to_csv(self.train_csv_path)

    def handle_file(
        self,
        idx: int,
        train_example_list: pd.DataFrame,
        filepath: str,
        hf: h5py.File,
        directory: str,
        min_z_score: float,
        before: int,
        after: int,
        crop_size: int,
        roi_size: int,
        output_h5_file: str,
        window_size: int = 50,
        stimulationframes: list[int] = [],
        n_frames: int = 1,
        foreground_background_split: float = 0.1,
    ):
        file = open_file(filepath)
        # find train examples with activity
        file_znorm = normalization.rolling_window_z_norm(file, window_size)
        if max(stimulationframes) >= file.shape[0]:
            print(
                f"WARNING: stimulationframes ({stimulationframes}) out of range of loaded file with number of frames ({file.shape[0]})."
            )
            stimulationframes = [
                stimframe
                for stimframe in stimulationframes
                if stimframe < file.shape[0]
            ]
        # will go through all frames and extract events that within a meaned kernel exceed the
        # min_z_score threshold
        # returns a list of events in the form [frame, y-coord, x-coord]
        frames_and_positions = get_frames_position(
            file_znorm,
            min_z_score,
            before,
            after,
            crop_size,
            roi_size,
            stimulationframes,
            n_frames,
            foreground_background_split,
        )

        if len(frames_and_positions) == 0:
            return idx

        mean = np.mean(file, axis=0)
        std = np.std(file, axis=0)
        file = normalization.z_norm(file, mean, std)

        # create dict to be stored as h5 file
        for event in frames_and_positions:
            target_frame, y_pos, x_pos = event
            train_example_list.append([str(idx), filepath, target_frame, y_pos, x_pos])
            example = file[
                target_frame,
                y_pos : y_pos + crop_size,
                x_pos : x_pos + crop_size,
            ]
            hf.create_dataset(str(idx), data=example)
            idx += 1

    def handle_file_memory_optimized(
        self,
        idx: int,
        train_example_list: pd.DataFrame,
        filepath: str,
        hf: h5py.File,
        directory: str,
        min_z_score: float,
        before: int,
        after: int,
        crop_size: int,
        roi_size: int,
        output_h5_file: str,
        window_size: int = 50,
        stimulationframes: list[int] = [],
        n_frames: int = 1,
        foreground_background_split: float = 0.1,
    ):
        file = open_file(filepath)
        if max(stimulationframes) >= file.shape[0]:
            print(
                f"WARNING: stimulationframes ({stimulationframes}) out of range of loaded file with number of frames ({file.shape[0]})."
            )
            stimulationframes = [
                stimframe
                for stimframe in stimulationframes
                if stimframe < file.shape[0]
            ]
        # -- numpy memmaps --
        mmap_file_path = os.path.join(directory, "mmap_time_file.npy")
        file_shape = file.shape
        np.save(mmap_file_path, file)
        # wrap memmap around file on disk
        mmap_file = np.memmap(
            mmap_file_path, dtype="float64", mode="w+", shape=file_shape
        )
        mmap_file[:] = file[:]
        # clear ram by removing file
        del file
        # flush mmap to disk
        mmap_file.flush()

        mmap_znorm_file_path = os.path.join(directory, "mmap_time_znorm_file.npy")
        # wrap memmap around file on disk
        mmap_znorm_file = np.memmap(
            mmap_znorm_file_path, dtype="float64", mode="w+", shape=file_shape
        )
        # find train examples with activity
        znorm_file = normalization.rolling_window_z_norm_memory_optimized(
            mmap_file, window_size, directory
        )
        mmap_znorm_file[:] = znorm_file[:]
        # flush mmap to disk
        del znorm_file
        mmap_znorm_file.flush()

        # will go through all frames and extract events that within a meaned kernel exceed the
        # min_z_score threshold
        # returns a list of events in the form [frame, y-coord, x-coord]
        frames_and_positions = get_frames_position(
            mmap_znorm_file,
            min_z_score,
            before,
            after,
            crop_size,
            roi_size,
            stimulationframes,
            n_frames,
            foreground_background_split,
        )

        print(f"Found {len(frames_and_positions)} example(s) in file {filepath}")

        if len(frames_and_positions) == 0:
            return idx

        mean = np.mean(mmap_file, axis=0)
        std = np.std(mmap_file, axis=0)

        mmap_file[:] = normalization.z_norm(mmap_file, mean, std)[:]

        # create dict to be stored as h5 file
        for event in frames_and_positions:
            target_frame, y_pos, x_pos = event
            train_example_list.append([str(idx), filepath, target_frame, y_pos, x_pos])
            example = mmap_file[
                target_frame,
                y_pos : y_pos + crop_size,
                x_pos : x_pos + crop_size,
            ]
            hf.create_dataset(str(idx), data=example)
            idx += 1
        return idx
