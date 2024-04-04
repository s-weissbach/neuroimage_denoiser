import os
import pandas as pd
import numpy as np
import h5py
from alive_progress import alive_bar
import deep_iglu_denoiser.utils.normalization as normalization
from deep_iglu_denoiser.utils.activitymap import (
    get_frames_position,
    get_frames_position_stimframes,
)
from deep_iglu_denoiser.utils.open_file import open_file


class TrainFiles:
    def __init__(
        self,
        fileendings: list[str],
        min_z_score: float,
        crop_size: int,
        roi_size: int,
        output_h5_file: str,
        min_z_score_activity: float = 1.5,
        window_size: int = 50,
        activitymap: bool = False,
        stimulationframes: list[int] = [],
        n_frames: int = 1,
        foreground_background_split: float = 0.1,
        overwrite: bool = False,
        pre_frames: int = 5,
        post_frames: int = 5
    ) -> None:
        """
        Initialize TrainFiles object.

        Args:
            train_csv_path (str): Path to the CSV file containing training examples information.
            overwrite (bool, optional): If True, overwrite existing files. Default is False.
        """
        self.fileendings = fileendings
        self.min_z_score = min_z_score
        self.min_z_score_activity = min_z_score_activity
        self.crop_size = crop_size
        self.roi_size = roi_size
        self.output_h5_file = output_h5_file
        self.window_size = window_size
        self.activitymap = activitymap
        self.stimulationframes = stimulationframes
        self.n_frames = n_frames
        self.foreground_background_split = foreground_background_split
        self.overwrite = overwrite
        self.pre_frames = pre_frames
        self.post_frames = post_frames
        self.file_list = {}

    def files_to_traindata(
        self,
        directory: str,
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
        files_to_do = []

        # find files recursively
        for root, _, files in os.walk(directory):
            for file in files:
                if not any([file.endswith(ending) for ending in self.fileendings]):
                    continue
                files_to_do.append(os.path.join(root, file))
        print(f"Found {len(files_to_do)} file(s).")

        if os.path.exists(self.output_h5_file) and not self.overwrite:
            print("Found existing h5-file. Will append.")
            # find index
            hf = h5py.File(self.output_h5_file, "a")
            self.idx = np.max([int(key) for key in hf.keys()]) + 1
            hf.close()
        elif os.path.exists(self.output_h5_file) and self.overwrite:
            os.remove(self.output_h5_file)
            # initalize h5 file
            hf = h5py.File(self.output_h5_file, "w")
            self.idx = 0
            hf.close()
        else:
            # initalize h5 file
            hf = h5py.File(self.output_h5_file, "w")
            self.idx = 0
            hf.close()

        with alive_bar(len(files_to_do)) as bar:
            for filepath in files_to_do:
                if self.activitymap:
                    self.handle_file_activitymap(filepath)
                else:
                    self.handle_file(filepath)
                bar()


    def handle_file(
        self,
        filepath: str,
    ) -> None:
        file = open_file(filepath)
        if len(file.shape) <= 2:
            print(f"WARNING: skipped ({filepath}), not a series.")
            return
        # find train examples with activity
        mean = np.mean(file, axis=0)
        std = np.std(file, axis=0)
        file = normalization.z_norm(file, mean, std)
        if max(self.stimulationframes) >= file.shape[0]:
            print(
                f"WARNING: stimulationframes ({self.stimulationframes}) out of range of loaded file with number of frames ({file.shape[0]})."
            )
        stimulationframes = [
            stimframe
            for stimframe in self.stimulationframes
            if stimframe < file.shape[0]
        ]
        # will go through all frames and extract events that within a meaned kernel exceed the
        # min_z_score threshold
        # returns a list of events in the form [frame, y-coord, x-coord]
        frames_and_positions = get_frames_position_stimframes(
            file,
            self.min_z_score,
            self.min_z_score_activity,
            self.crop_size,
            self.roi_size,
            stimulationframes,
            self.n_frames,
            self.foreground_background_split,
        )
        print(f"Found {len(frames_and_positions)} example(s) in file {filepath}")
        if len(frames_and_positions) == 0:
            return

        hf = h5py.File(self.output_h5_file, "a")
        # create dict to be stored as h5 file
        for event in frames_and_positions:
            target_frame, y_pos, x_pos = event
            from_frame = target_frame-self.pre_frames
            to_frame = target_frame+self.post_frames+1
            if from_frame < 0 or to_frame > file.shape[0]: continue
            example = file[
                from_frame:to_frame,
                y_pos : y_pos + self.crop_size,
                x_pos : x_pos + self.crop_size,
            ]
            hf.create_dataset(str(self.idx), data=example)
            self.idx += 1
        hf.close()

    def handle_file_activitymap(
        self,
        filepath: str,
    ) -> None:
        file = open_file(filepath)
        if len(file.shape) <= 2:
            print(f"WARNING: skipped ({filepath}), not a series.")
            return
        # find train examples with activity
        file_znorm = normalization.rolling_window_z_norm(file, self.window_size)
        # will go through all frames and extract events that within a meaned kernel exceed the
        # min_z_score threshold
        # returns a list of events in the form [frame, y-coord, x-coord]
        frames_and_positions = get_frames_position(
            file_znorm,
            self.min_z_score,
            self.crop_size,
            self.roi_size,
            self.foreground_background_split,
        )
        print(f"Found {len(frames_and_positions)} example(s) in file {filepath}")
        if len(frames_and_positions) == 0:
            return
        mean = np.mean(file, axis=0)
        std = np.std(file, axis=0)
        file = normalization.z_norm(file, mean, std)
        hf = h5py.File(self.output_h5_file, "a")
        # create dict to be stored as h5 file
        for event in frames_and_positions:
            target_frame, y_pos, x_pos = event
            from_frame = target_frame-self.pre_frames
            to_frame = target_frame+self.post_frames+1
            if from_frame < 0 or to_frame > file.shape[0]: continue
            example = file[
                from_frame:to_frame,
                y_pos : y_pos + self.crop_size,
                x_pos : x_pos + self.crop_size,
            ]
            hf.create_dataset(str(self.idx), data=example)
            self.idx += 1
        hf.close()
