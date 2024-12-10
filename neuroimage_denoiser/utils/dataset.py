import numpy as np
import torch
import h5py
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset


class DataSet(Dataset):
    def __init__(
        self,
        train_h5: str,
        batch_size: int,
        noise_center: float = 0,
        noise_scale: float = 1.5,
        apply_gaussian_filter: bool = False,
        sigma_gaussian_filter: float = 1.0,
    ):
        """
        Initialize the dataset with HDF5 file and optional noise parameters.

        Args:
            train_h5 (str): Path to the HDF5 file containing training samples.
            noise_center (float, optional): Center of the noise distribution. Default is 0.
            noise_scale (float, optional): Scale of the noise distribution. Default is 1.5.
            apply_gaussian_filter (bool, optional): Whether to apply Gaussian filtering to targets.
            sigma_gaussian_filter (float, optional): Sigma for Gaussian filter if applied.
        """
        self.h5_file = h5py.File(train_h5, "r")
        self.samples = list(self.h5_file.keys())
        self.noise_center = noise_center
        self.noise_scale = noise_scale
        self.apply_gaussian_filter = apply_gaussian_filter
        self.sigma_gaussian_filter = sigma_gaussian_filter
        print(f"Found {len(self.samples)} samples to train.")

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.samples)

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

    def add_gaussian_noise(self, arr: np.ndarray) -> np.ndarray:
        """
        Add gaussian noise to an image for training.

        Args:
            arr (np.ndarray): original array to add noise to

        Returns:
            np.ndarray: array with added noise
        """
        noise = np.random.normal(self.noise_center, self.noise_scale, size=arr.shape)
        return np.add(arr, noise)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Args:
            idx (int): Index of the sample to get

        Returns:
            tuple: (input_tensor, target_tensor) pair
        """
        # Get sample
        sample = np.array(self.h5_file[self.samples[idx]])

        # Create input with noise
        X = self.add_gaussian_noise(sample.copy())
        X = torch.tensor(X, dtype=torch.float).reshape(1, X.shape[0], X.shape[1])

        # Process target
        if self.apply_gaussian_filter:
            sample = gaussian_filter(sample, self.sigma_gaussian_filter)
        y = torch.tensor(sample, dtype=torch.float).reshape(
            1, sample.shape[0], sample.shape[1]
        )
        return X, y

    def __del__(self):
        """Cleanup the HDF5 file handle."""
        self.h5_file.close()
