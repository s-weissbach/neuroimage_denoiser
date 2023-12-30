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
