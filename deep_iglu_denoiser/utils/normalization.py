import numpy as np
from scipy.ndimage import uniform_filter1d


def z_norm(img: np.ndarray[np.float64], mean: np.ndarray[np.float64], std: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
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

def moving_std(img: np.ndarray[np.float64], start:int, end:int) -> np.ndarray[np.float64]:
    """
    Calculate the moving standard deviation of a numpy array within a specified range.

    Args:
        arr (np.ndarray[np.float64]): Input array.
        start (int): Starting index of the range.
        end (int): Ending index of the range.

    Returns:
        np.ndarray[np.float64]: Moving standard deviation.
    """
    std = np.std(img[start:end], axis=0)
    return std

def rolling_window_z_norm(img: np.ndarray[np.int64], window_size: int) -> np.ndarray[np.float64]:
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
    mean = uniform_filter1d(img,window_size,axis=0,mode='constant')
    std = []
    for idx in range(img.shape[0]):
        start = max(0, idx - before)
        end = min(img.shape[0] - 1, idx + after)
        std.append(moving_std(img, start, end))
    normed_img = np.divide(np.subtract(img, mean), std)
    return normed_img


def reverse_z_norm(img: np.ndarray[np.float64], mean: np.ndarray[np.float64], std: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
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
