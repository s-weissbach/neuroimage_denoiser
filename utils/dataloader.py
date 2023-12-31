from utils.trainfiles import TrainFiles
import numpy as np
import tifffile
import torch
import utils.normalization as normalization


class DataLoader:
    def __init__(
        self,
        t_files: TrainFiles,
        batch_size: int,
        n_pre: int,
        n_post: int,
        train_height: int,
        train_width: int,
        load_multiple_targets_per_file: bool = False,
        n_multiple_targets: int = 5,
        use_active_frames: bool = True,
        crop_size: int = 16,
    ):
        """
        Initialize the DataLoader.

        Parameters:
        - t_files: TrainFiles object containing file information.
        - batch_size: Number of samples in each batch.
        - n_pre: Number of frames before the target frame.
        - n_post: Number of frames after the target frame.
        - train_height: Height to which images will be cropped or padded.
        - train_width: Width to which images will be cropped or padded.
        - load_multiple_targets_per_file: Whether to load multiple targets from a file.
        - n_multiple_targets: Number of targets to load per file.
        - max_intensity: Maximum intensity value in the images.
        """
        if load_multiple_targets_per_file and n_multiple_targets > batch_size:
            raise ValueError(
                f"Invalid configuration: n_multiple_targets ({n_multiple_targets}) must be smaller or equal to batch_size ({batch_size})."
            )
        elif load_multiple_targets_per_file and batch_size % n_multiple_targets != 0:
            raise ValueError(
                f"Invalid configuration: batch_size ({batch_size}) must be divisible by n_multiple_targets ({n_multiple_targets})."
            )
        np.random.seed(42)
        self.file_dict = t_files.file_dict
        self.batch_size = batch_size
        self.n_pre = n_pre
        self.n_post = n_post
        self.train_height = train_height
        self.train_width = train_width
        self.load_multiple_targets_per_file = load_multiple_targets_per_file
        self.use_active_frames = use_active_frames
        self.crop_size = crop_size
        self.epoch_done = False
        if self.load_multiple_targets_per_file:
            self.n_multiple_targets = n_multiple_targets
        else:
            self.n_multiple_targets = 1
        self.train_examples = []
        self.norm_vals = {}
        for iglu_movie in self.file_dict.keys():
            path = self.file_dict[iglu_movie]["filepath"]
            movie_len = self.file_dict[iglu_movie]["shape"][0]
            active_frames = self.file_dict[iglu_movie]["frames_and_positions"]
            if self.load_multiple_targets_per_file:
                iterate_to = movie_len - self.n_post - 1 - n_multiple_targets
                iterate_step = self.n_multiple_targets
            else:
                iterate_to = movie_len - self.n_post - 1
                iterate_step = 1
            if use_active_frames:
                self.train_examples += [
                    [path, frame[0], frame[1], frame[2]]
                    for frame in active_frames
                    if (frame[0] > self.n_pre and frame[0] <= iterate_to)
                ]
            else:
                self.train_examples += [
                    [path, target, np.nan, np.nan]
                    for target in range(self.n_pre + 1, iterate_to, iterate_step)
                ]
            self.norm_vals[path] = {
                "mean": self.file_dict[iglu_movie]["mean"],
                "std": self.file_dict[iglu_movie]["std"],
            }
        self.shuffle_array()
        self.X_list = []
        self.y_list = []

    def shuffle_array(self) -> None:
        """
        Shuffle the training examples for a new epoch.
        """
        self.epoch_done = False
        random_order = np.arange(len(self.train_examples))
        np.random.shuffle(random_order)
        self.available_train_examples = [
            self.train_examples[idx] for idx in random_order
        ]

    def crop_pad_img(
        self, img: np.ndarray, y_min: int = -1, x_min: int = -1
    ) -> np.ndarray:
        """
        Shuffle the training examples for a new epoch.
        """
        # in case of training on small patches, cropping will work for sure (as it is insured
        # by trainfiles.py). So it can directly return the croped images
        if self.use_active_frames:
            img = img[
                :, y_min : y_min + self.train_height, x_min : x_min + self.train_width
            ]
            return img
        # else the input image is cropped/padded
        _, height, width = img.shape
        diff_height = self.train_height - height
        diff_width = self.train_width - width
        pad = False
        # crop
        if diff_height < 0:
            start = np.random.choice(abs(diff_height))
            stop = height - (abs(diff_height) - start)
            img = img[:, start:stop, :]
        if diff_width < 0:
            start = np.random.choice(abs(diff_width))
            stop = width - (abs(diff_width) - start)
            img = img[:, :, start:stop]
        # pad
        if diff_height > 0:
            pad = True
            up_pad = diff_height // 2
            down_pad = diff_height - diff_height // 2
        else:
            up_pad = 0
            down_pad = 0
        if diff_width > 0:
            pad = True
            left_pad = diff_width // 2
            right_pad = diff_width - diff_width // 2
        else:
            left_pad = 0
            right_pad = 0
        if pad:
            pad_coords = ((0, 0), (up_pad, down_pad), (left_pad, right_pad))
            img = np.pad(
                img,
                pad_coords,  # type: ignore
                mode="reflect",
            )
        return img

    def create_train_example(
        self,
        filepath: str,
        target: int,
        mean: float,
        std: float,
        y_min: int,
        x_min: int,
    ) -> None:
        """
        Create a training example using the specified file, target, mean, and std.

        Parameters:
        - filepath: Filepath of the image.
        - target: Target frame index.
        - mean: Mean value for scaling.
        - std: Standard deviation for scaling.
        """
        img = tifffile.imread(filepath)
        img = self.crop_pad_img(img, y_min, x_min)
        for i in range(self.n_multiple_targets):
            target_ = target + i
            # Extract frames, scale, and copy middle frame to y and delete
            X_tmp = normalization.z_norm(
                img[target_ - self.n_pre - 1 : target_ + self.n_post], mean, std
            )
            y_tmp = (
                X_tmp[self.n_pre].copy().reshape(1, self.train_height, self.train_width)
            )
            X_tmp = np.delete(X_tmp, self.n_pre, axis=0)
            self.X_list.append(X_tmp)
            self.y_list.append(y_tmp)

    def get_batch(self) -> bool:
        """
        Get a batch of training examples.

        Returns:
        - True if a batch is successfully created, False if the epoch is done.
        """
        self.X_list = []
        self.y_list = []
        for _ in range(self.batch_size // self.n_multiple_targets):
            if len(self.available_train_examples) == 0:
                # Check if available_train_examples is empty, indicating the end of the epoch
                self.epoch_done = True
                return False
            filepath, target, y_min, x_min = self.available_train_examples.pop(0)
            mean = self.norm_vals[filepath]["mean"]
            std = self.norm_vals[filepath]["std"]
            self.create_train_example(filepath, target, mean, std, y_min, x_min)

        self.X = torch.tensor(np.array(self.X_list), dtype=torch.float)
        del self.X_list
        self.y = torch.tensor(np.array(self.y_list), dtype=torch.float)
        del self.y_list
        return True
