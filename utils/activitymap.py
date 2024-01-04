import numpy as np
import ctypes
from multiprocessing import Array, Pool

def get_patch_value(patch: np.ndarray, roi_size: int) -> float:
    """
    Calculate the maximum mean value within a sliding window (patch) of the specified size.
    Patch would represent a part of the image on that is trained, while ROI should roughly
    represent the size of one ROI (e.g. synapse).

    Parameters:
    - patch (np.ndarray): Input patch as a 2D NumPy array.
    - roi_size (int): Size of the sliding window (Region of Interest).

    Returns:
    - float: Maximum mean value within the specified patch.
    """
    max_y,max_x = patch.shape
    max_val = -np.inf
    for y in range(max_y-roi_size):
        for x in range(max_x-roi_size):
            mean = np.mean(patch[y:y+roi_size,x:x+roi_size])
            if mean > max_val:
                max_val = mean
    return max_val


def transform_frame(parameters: tuple) -> list[float]:
    frame_idx, kernelsize, h_activitymap, w_activitymap, roi_size = parameters
    frame = shared_array[frame_idx]
    activity_map_frame = []
    for y in range(h_activitymap):
        start_y = y * kernelsize
        stop_y = (y + 1) * kernelsize
        row = []
        for x in range(w_activitymap):
            start_x = x * kernelsize
            stop_x = (x + 1) * kernelsize
            row.append(get_patch_value(frame[start_y:stop_y, start_x:stop_x], roi_size))
        activity_map_frame.append(row)
    return activity_map_frame


def compute_activitymap(img: np.ndarray, kernelsize: int, roi_size: int, n_threads: int) -> np.ndarray:
    """
    Compute the activity map for a sequence of image frames by applying the
    transform_frame function.

    Parameters:
    - img (np.ndarray): Input image sequence as a 3D NumPy array.
    - kernelsize: Size of the kernel used for patch extraction (image size on that is trained).
    - roi_size: Size of the sliding window (Region of Interest).

    Returns:
    - np.ndarray: Activity map for the input image sequence.
    """
    h_activitymap = img.shape[1] // kernelsize
    w_activitymap = img.shape[2] // kernelsize
    shared_array_base = Array(ctypes.c_double, img.shape[0]*img.shape[1]*img.shape[2])
    global shared_array
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array = shared_array.reshape(img.shape)
    for i,frame in enumerate(img):
        shared_array[i] = frame
    parameters = []
    for frame_idx in range(img.shape[0]):
        parameters.append((frame_idx, kernelsize, h_activitymap, w_activitymap, roi_size))
    with Pool(n_threads) as pool:
        activitymap = list(pool.imap(transform_frame,parameters))
    return np.array(activitymap)


def get_frames_position(
    img: np.ndarray,
    min_z_score: float,
    kernelsize: int = 32,
    roi_size: int = 4,
    foreground_background_split: float = 0.5,
    n_threads: int = 1
) -> list[list[int]]:
    """
    Identify positions of frames based on the computed activity map and a minimum Z-score threshold.

    Parameters:
    - img (np.ndarray): Input image sequence as a 3D NumPy array.
    - min_z_score (float): Minimum Z-score threshold for identifying frames.
    - kernelsize (int): Size of the kernel used for patch extraction.
    - roi_size (int): Size of the sliding window (Region of Interest).
    - foreground_background_split (float): Split ratio between foreground and background.

    Returns:
    - list[list[int]]: List of frame positions, each represented as [frame_index, y_position, x_position].
    """
    frames_w_pos = []
    activitymap = compute_activitymap(img, kernelsize, roi_size, n_threads)
    above_z = np.argwhere(activitymap > min_z_score)
    for example in above_z:
        frame,y,x = example
        frames_w_pos.append(
            [int(frame), int(y * kernelsize), int(x * kernelsize)]
        )
    bg_images_to_select = (1 / foreground_background_split - 1) * len(frames_w_pos)
    below_z = np.argwhere(activitymap <= min_z_score)
    np.random.shuffle(below_z)
    for i, example in enumerate(below_z):
        if i > bg_images_to_select:
            break
        frame,y,x = example
        frames_w_pos.append(
            [int(frame), int(y * kernelsize), int(x * kernelsize)]
        )
    return frames_w_pos
