import numpy as np
from typing import Union
import ctypes
from multiprocessing import Array, Pool


def z_norm(
    img: np.ndarray, mean: np.ndarray, std: np.ndarray
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



def moving_mean(parameters: tuple[int]) -> np.ndarray:
    start,end = parameters
    mean = np.mean(shared_array[start:end],axis=0)
    return mean

def moving_std(parameters: tuple[int]) -> np.ndarray:
    start,end = parameters
    std = np.std(shared_array[start:end],axis=0)
    return std

def rolling_window_z_norm(img: np.ndarray, window_size: int, n_threads: int = 1):
    # make img a shared array for multiprocessing
    shared_array_base = Array(ctypes.c_double, img.shape[0]*img.shape[1]*img.shape[2])
    global shared_array
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array = shared_array.reshape(img.shape)
    for i,frame in enumerate(img):
        shared_array[i] = frame
    before = window_size // 2
    after = window_size - before
    parameters = []
    for idx in range(img.shape[0]):
        start = max(0, idx-before)
        end = min(img.shape[0]-1,idx+after)
        parameters.append((start,end))
    with Pool(n_threads) as pool:
        mean = np.array(list(pool.imap(moving_mean,parameters)))
        std = np.array(list(pool.imap(moving_std,parameters)))
    del shared_array
    del shared_array_base
    normed_img = np.divide(np.subtract(img, mean), std)
    return normed_img

def reverse_z_norm(
    img: np.ndarray, mean: np.ndarray, std: np.ndarray
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
