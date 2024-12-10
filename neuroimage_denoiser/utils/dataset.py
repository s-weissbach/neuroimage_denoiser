import numpy as np
import torch
import h5py
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset


class DataSet(Dataset):
    def __init__(
        self,
        train_h5: str,
        noise_center: float = 0,
        noise_scale: float = 1.5,
        apply_gaussian_filter: bool = False,
        sigma_gaussian_filter: float = 1.0,
        num_frames: int = 5,
        min_start_frame: int = -1,
        max_start_frame: int = -1,
    ):
        """
        Initialize the dataset with HDF5 file, batch size, and optional noise parameters.

        Args:
            train_h5 (str): Path to the HDF5 file containing training samples.
            batch_size (int): Number of samples in each batch.
            noise_center (float, optional): Center of the noise distribution. Default is 0.
            noise_scale (float, optional): Scale of the noise distribution. Default is 1.5.
        """
        self.h5_file = h5py.File(train_h5, "r")
        self.samples = list(self.h5_file.keys())
        self.noise_center = noise_center
        self.noise_scale = noise_scale
        self.apply_gaussian_filter = apply_gaussian_filter
        self.sigma_gaussian_filter = sigma_gaussian_filter
        self.num_frames = num_frames
        self.min_start_frame = 0 if min_start_frame == -1 else min_start_frame
        self.max_start_frame = max_start_frame
        print(f"Found {len(self.samples)} samples to train.")

    def __len__(self) -> int:
        return len(self.samples)

    def add_gaussian_noise(self, arr: np.ndarray) -> np.ndarray:
        """
        Add gaussian noise to an image or image sequence for training.

        Parameters:
        - arr (np.ndarray): original array to that the noise should be added

        Returns:
        - arr (np.ndarray): array with added noise
        """
        noise = np.random.normal(self.noise_center, self.noise_scale, size=arr.shape)
        return np.add(arr, noise)

    def get_frame_sequence(self, sequence: np.ndarray) -> np.ndarray:
        max_start = sequence.shape[0] - self.num_frames
        if self.max_start_frame > 0:
            max_start = min(max_start, self.max_start_frame)
        if max_start > 0:
            start_idx = np.random.randint(self.min_start_frame, max_start)
            sequence = sequence[start_idx : start_idx + self.num_frames]
        return sequence

    def __getitem__(self, idx):
        sequence = np.array(self.h5_file[self.samples[idx]])
        sequence = self.get_frame_sequence(sequence)
        # Create input with noise
        X = self.add_gaussian_noise(sequence.copy())
        X = torch.tensor(X, dtype=torch.float).reshape(
            1, X.shape[0], X.shape[1], X.shape[2]
        )
        # Process target
        if self.apply_gaussian_filter:
            sequence = gaussian_filter(sequence, self.sigma_gaussian_filter)
        y = torch.tensor(sequence, dtype=torch.float).reshape(
            1, sequence.shape[0], sequence.shape[1], sequence.shape[2]
        )

        return X, y

    def __del__(self):
        self.h5_file.close()
