import numpy as np


def z_norm(img: np.ndarray, mean: float, std: float) -> np.ndarray:
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


def reverse_z_norm(img: np.ndarray, mean: float, std: float) -> np.ndarray:
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
