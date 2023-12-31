import numpy as np
from typing import Union


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


def rolling_window_z_norm(
    img: np.ndarray,
    window_size: int,
) -> np.ndarray:
    frames_before = window_size // 2
    frames_after = window_size - frames_before
    z_scaled_img = []
    for idx_frame in range(img.shape[0]):
        start = max(0, idx_frame - frames_before)
        end = min(img.shape[0] - 1, idx_frame + frames_after + 1)
        img_current_window = img[start:end]
        mean = np.mean(img_current_window)
        std = np.std(img_current_window)
        z_scaled_img.append(z_norm(img[idx_frame], mean, std))
    return np.array(z_scaled_img)


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
