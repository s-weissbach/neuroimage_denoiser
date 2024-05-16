import numpy as np
import torch
import h5py


class DataLoader:
    def __init__(
        self, 
        train_h5: str,
        batch_size: int,
        n_frames: int
    ):
        """
        Initialize the DataLoader class for loading training data from an HDF5 file.

        Parameters:
        - train_h5 (str): Path to the HDF5 file containing training data.
        - batch_size (int): Size of each batch during training.
        - n_frames (int): Number of frames around the target frame.

        Attributes:
        - h5_file (h5py.File): HDF5 file object for accessing training data.
        - train_samples (list): List of keys in the HDF5 file representing training samples.
        - batch_size (int): Size of each batch during training.
        - pre_frames (int): Number of frames before the target frame in each sample.
        - post_Frames (int): Number of frames after the target frame in each sample.
        - epoch_done (bool): Flag indicating whether the current epoch is completed.
        - available_train_examples (list): List of available training examples for shuffling.
        - X_list (list): List to store input frames for a batch.
        - y_list (list): List to store target frames for a batch.
        """
        np.random.seed(42)
        self.h5_file = h5py.File(train_h5, "r")
        self.train_samples = list(self.h5_file.keys())
        self.batch_size = batch_size
        self.pre_frames = n_frames // 2
        self.post_Frames = n_frames - self.pre_frames
        self.epoch_done = False
        print(
            f"Found {len(self.train_samples)} samples to train. \n Batch size is {self.batch_size} -> {len(self.train_samples)//self.batch_size} iterations per epoch."
        )
        self.shuffle_array()
        self.X_list = []
        self.y_list = []

    def __len__(self) -> int:
        """
        Returns the number of train samples in the dataset.

        Returns:
            int: Number of train samples in the dataset.
        """
        return len(self.train_samples) // self.batch_size

    def shuffle_array(self) -> None:
        """
        Shuffle the training examples for a new epoch.
        """
        self.epoch_done = False
        random_order = np.arange(len(self.train_samples))
        np.random.shuffle(random_order)
        self.available_train_examples = [
            self.train_samples[idx] for idx in random_order
        ]

    def get_batch(self) -> bool:
        """
        Get a batch of training examples.

        Returns:
        - True if a batch is successfully created, False if the epoch is done.
        """
        self.X_list = []
        self.y_list = []
        for _ in range(self.batch_size):
            if len(self.available_train_examples) == 0:
                # Check if available_train_examples is empty, indicating the end of the epoch
                self.epoch_done = True
                return False
            h5_idx = self.available_train_examples.pop(0)
            X_tmp = np.array(self.h5_file.get(h5_idx))
            y_tmp = (
                X_tmp[self.pre_frames, :, :]
                .copy()
                .reshape(1, X_tmp.shape[1], X_tmp.shape[2])
            )
            X_tmp = np.delete(X_tmp, self.pre_frames, axis=0)
            self.y_list.append(y_tmp)
            self.X_list.append(X_tmp)
        self.X = torch.tensor(np.array(self.X_list), dtype=torch.float)
        self.X_list = []
        self.y = torch.tensor(np.array(self.y_list), dtype=torch.float)
        self.y_list = []
        return True
