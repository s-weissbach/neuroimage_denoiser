import numpy as np
from typing import Union
from multiprocessing import Pool


def z_norm(
    img: np.ndarray, mean: Union[float, np.float64], std: Union[float, np.float64]
) -> np.ndarray:
    """
    Z-scaling for the image. z = (x-µ)/σ

    Parameters:
    - img: Input image.
    - mean: Mean value for scaling.
    - std: Standard deviation for scaling.

    Returns:
    - z-scaled image.
    """
    return np.divide(np.subtract(img, mean), std)


def mp_rolling_window_z_norm(parameters: tuple) -> list[np.ndarray]:
    img, seq_start, seq_end, frames_before, frames_after = parameters
    result = []
    for idx_frame in range(seq_start, seq_end):
        start = max(0, idx_frame - frames_before)
        end = min(img.shape[0] - 1, idx_frame + frames_after + 1)
        img_current_window = img[start:end]
        mean = np.mean(img_current_window)
        std = np.std(img_current_window)
        result.append(z_norm(img[idx_frame], mean, std))
    return result


def rolling_window_z_norm(
    img: np.ndarray, window_size: int, n_threads: int = 10
) -> np.ndarray:
    frames_before = window_size // 2
    frames_after = window_size - frames_before
    z_scaled_img = []
    parameters = []
    seq_len = img.shape[0]
    for start in range(0, seq_len, seq_len // n_threads):
        end = start + seq_len // n_threads
        parameters.append([img, start, end, frames_before, frames_after])
    parameters[-1][-1] = seq_len
    with Pool(n_threads) as pool:
        z_scaled_img = list(pool.imap(mp_rolling_window_z_norm, parameters))
    return np.array(z_scaled_img).reshape(img.shape)


def reverse_z_norm(
    img: np.ndarray, mean: Union[float, np.float64], std: Union[float, np.float64]
) -> np.ndarray:
    """
    Reverse z-scaling for the image. x = z*σ+µ

    Parameters:
    - img: Input image.
    - mean: Mean value for scaling.
    - std: Standard deviation for scaling.

    Returns:
    - reversed z-scored image.
    """
    return np.add(np.multiply(img, std), mean)
