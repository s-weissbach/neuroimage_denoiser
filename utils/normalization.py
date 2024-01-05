import numpy as np
import ctypes
from multiprocessing import Array, Pool


def z_norm(img: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Pixelwise z-scaling for the image. z = (x-µ)/σ

    Parameters:
    - img: Input image.
    - mean: Mean matrix for scaling
    - std: Standard deviation matrix for scaling.

    Returns:
    - z-scaled image.
    """
    return np.divide(np.subtract(img, mean), std)


def moving_mean(parameters: tuple[int, int]) -> np.ndarray:
    """
    Calculate the moving mean for a specified range of frames.

    Parameters:
    - parameters: Tuple containing the start and end indices.

    Returns:
    - np.ndarray: Moving mean values.
    """
    start, end = parameters
    mean = np.mean(shared_array[start:end], axis=0)
    return mean


def moving_std(parameters: tuple[int, int]) -> np.ndarray:
    """
    Calculate the moving standard deviation for a specified range of frames.

    Parameters:
    - parameters: Tuple containing the start and end indices.

    Returns:
    - np.ndarray: Moving standard deviation values.
    """
    start, end = parameters
    std = np.std(shared_array[start:end], axis=0)
    return std


def rolling_window_z_norm(img: np.ndarray, window_size: int, n_threads: int = 1):
    """
    Apply rolling window z-scaling to an image sequence using multiprocessing.

    Parameters:
    - img: Input image sequence.
    - window_size: Size of the rolling window.
    - n_threads: Number of threads for parallel processing (default is 1).

    Returns:
    - np.ndarray: Z-scaled image sequence.
    """
    # make img a shared array for multiprocessing
    shared_array_base = Array(
        ctypes.c_double, img.shape[0] * img.shape[1] * img.shape[2]
    )
    global shared_array
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array = shared_array.reshape(img.shape)
    for i, frame in enumerate(img):
        shared_array[i] = frame
    before = window_size // 2
    after = window_size - before
    parameters = []
    for idx in range(img.shape[0]):
        start = max(0, idx - before)
        end = min(img.shape[0] - 1, idx + after)
        parameters.append((start, end))
    with Pool(n_threads) as pool:
        mean = np.array(list(pool.imap(moving_mean, parameters)))
        std = np.array(list(pool.imap(moving_std, parameters)))
    del shared_array
    del shared_array_base
    normed_img = np.divide(np.subtract(img, mean), std)
    return normed_img


def reverse_z_norm(img: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Reverse z-scaling for the image. x = z*σ+µ

    Parameters:
    - img: Input image.
    - mean: Mean matrix for scaling.
    - std: Standard deviation matrix for scaling.

    Returns:
    - reversed z-scored image.
    """
    return np.add(np.multiply(img, std), mean)
