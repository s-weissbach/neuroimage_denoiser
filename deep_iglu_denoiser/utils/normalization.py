import numpy as np
import os
from scipy.ndimage import uniform_filter1d


def z_norm(
    img: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """
    Pixelwise z-scaling for the image. z = (x-µ)/σ

    Parameters:
    - img (np.ndarray[np.float64]): Input image.
    - mean (np.ndarray[np.float64]): Mean matrix for scaling
    - std (np.ndarray[np.float64]): Standard deviation matrix for scaling.

    Returns:
        np.ndarray[np.float64]: z-scaled image.
    """
    return np.divide(np.subtract(img, mean), std)


def moving_std(img: np.ndarray, start: int, end: int) -> np.ndarray:
    """
    Calculate the moving standard deviation of a numpy array within a specified range.

    Args:
        arr (np.ndarray[np.float64]): Input array.
        start (int): Starting index of the range.
        end (int): Ending index of the range.

    Returns:
        np.ndarray[np.float64]: Moving standard deviation.
    """
    return np.std(img[start:end], axis=0)


def rolling_window_z_norm(
    img: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """
    Apply rolling window z-scaling to an image sequence.

    Parameters:
    - img (np.ndarray[np.float64]): Input image sequence.
    - window_size (int): Size of the rolling window.

    Returns:
        np.ndarray[np.float64]: Z-scaled image sequence.
    """
    before = window_size // 2
    after = window_size - before
    mean = uniform_filter1d(img, window_size, axis=0, mode="constant")
    std = []
    for idx in range(img.shape[0]):
        start = max(0, idx - before)
        end = min(img.shape[0] - 1, idx + after)
        std.append(moving_std(img, start, end))
    return np.divide(np.subtract(img, mean), std)


def rolling_window_z_norm_memory_optimized(
    img: np.ndarray,
    window_size: int,
    base_dir: str,
) -> np.ndarray:
    """
    Apply rolling window z-scaling to an image sequence.

    Parameters:
    - img (np.ndarray[np.float64]): Input image sequence.
    - window_size (int): Size of the rolling window.

    Returns:
        np.ndarray[np.float64]: Z-scaled image sequence.
    """
    std_mmap = np.memmap(
        os.path.join(base_dir, "mmap_std.npy"),
        dtype="float64",
        mode="w+",
        shape=img.shape,
    )
    mean_mmap = np.memmap(
        os.path.join(base_dir, "mmap_mean.npy"),
        dtype="float64",
        mode="w+",
        shape=img.shape,
    )
    before = window_size // 2
    after = window_size - before
    mean_mmap[:] = uniform_filter1d(img, window_size, axis=0, mode="constant")[:]
    for idx in range(img.shape[0]):
        start = max(0, idx - before)
        end = min(img.shape[0] - 1, idx + after)
        std_mmap[idx] = moving_std(img, start, end)
    return np.divide(np.subtract(img, mean_mmap), std_mmap)


def reverse_z_norm(
    img: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """
    Reverse z-scaling for the image. x = z*σ+µ

    Parameters:
    - img (np.ndarray[np.float64]): Input image.
    - mean (np.ndarray[np.float64]): Mean matrix for scaling.
    - std (np.ndarray[np.float64]): Standard deviation matrix for scaling.

    Returns:
        np.ndarray[np.float64]; reversed z-scored image.
    """
    return np.add(np.multiply(img, std), mean)
