import numpy as np
import torch
import h5py


class DataLoader:
    def __init__(self, train_h5: str, batch_size: int, target_frame: int):
        """
        Initialize the DataLoader.

        Parameters:
        - train_h5 (str): Path to h5 train file (generated by trainfiles.py).
        - batch_size (int): Number of samples in each batch.
        - target_frame (int): Frame in sequence that should be predcited
        """
        np.random.seed(42)
        self.h5_file = h5py.File(train_h5, "r")
        self.train_samples = list(self.h5_file.keys())
        self.batch_size = batch_size
        self.target_frame = target_frame
        self.epoch_done = False
        print(
            f"Found {len(self.train_samples)} samples to train. \n Batch size is {self.batch_size} -> {len(self.train_samples)//self.batch_size} iterations per epoch."
        )
        self.shuffle_array()
        self.X_list = []
        self.y_list = []

    def __len__(self) -> int:
        return len(self.train_samples)

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
            self.y_list.append(
                X_tmp[self.target_frame].reshape(1, X_tmp.shape[1], X_tmp.shape[2])
            )
            X_tmp = np.delete(X_tmp, self.target_frame, axis=0)
            self.X_list.append(X_tmp)
        self.X = torch.tensor(np.array(self.X_list), dtype=torch.float)
        self.X_list = []
        self.y = torch.tensor(np.array(self.y_list), dtype=torch.float)
        self.y_list = []
        return True