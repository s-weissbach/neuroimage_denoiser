import numpy as np


def uint_to_float(img: np.ndarray) -> np.ndarray:
    """
    Convert an input array of unsigned integers to float64.

    Parameters:
    - img (np.ndarray): Input array containing unsigned integers.

    Returns:
    - np.ndarray: Output array with values converted to float64.
    """
    return img.astype(np.float64)


def float_to_uint(img: np.ndarray) -> np.ndarray:
    """
    Convert an input array of floats to unsigned integers (uint16),
    handling underflows and overflows.

    Parameters:
    - img (np.ndarray): Input array containing float64 values.

    Returns:
    - np.ndarray: Output array with values converted to uint16.
    """
    # handle underflows
    frames, y_coords, x_coords = np.where(img < 0)
    for frame, y, x in zip(frames, y_coords, x_coords):
        img[frame, y, x] = 0.0
    # handle overflows
    frames, y_coords, x_coords = np.where(img > 65535)
    for frame, y, x in zip(frames, y_coords, x_coords):
        img[frame, y, x] = 65535.0
    return img.astype(np.uint16)
