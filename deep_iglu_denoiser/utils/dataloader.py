import numpy as np
import torch
import h5py


class DataLoader:
    def __init__(
        self,
        train_h5: str,
        batch_size: int,
        pre_frames: int = 5,
        post_frames: int = 5
    ):
        """
        Initialize the dataset with HDF5 file, batch size, and optional noise parameters.

        Args:
            train_h5 (str): Path to the HDF5 file containing training samples.
            batch_size (int): Number of samples in each batch.
            noise_center (float, optional): Center of the noise distribution. Default is 0.
            noise_scale (float, optional): Scale of the noise distribution. Default is 1.5.
        """
        np.random.seed(42)
        self.h5_file = h5py.File(train_h5, "r")
        self.train_samples = list(self.h5_file.keys())
        self.batch_size = batch_size
        self.pre_frames = pre_frames
        self.post_Frames = post_frames
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
            y_tmp = X_tmp[self.pre_frames,:,:].copy().reshape(1, y_tmp.shape[0], y_tmp.shape[1])
            X_tmp = np.delete(X_tmp,self.n_pre,axis=0)
            self.y_list.append(y_tmp)
            self.X_list.append(self.add_gausian_noise(y_tmp.copy()))
        self.X = torch.tensor(np.array(self.X_list), dtype=torch.float)
        self.X_list = []
        self.y = torch.tensor(np.array(self.y_list), dtype=torch.float)
        self.y_list = []
        return True
