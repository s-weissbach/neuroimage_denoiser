import numpy as np


def transform_frame(
    img: np.ndarray, kernelsize: int, h_actmap: int, w_actmap: int
) -> list[float]:
    activity_map_frame = []
    for y in range(h_actmap):
        start_y = y * kernelsize
        stop_y = (y + 1) * kernelsize
        row = []
        for x in range(w_actmap):
            start_x = x * kernelsize
            stop_x = (x + 1) * kernelsize
            row.append(np.mean(img[start_y:stop_y, start_x:stop_x]))
        activity_map_frame.append(row)
    return activity_map_frame


def compute_activitymap(img: np.ndarray, kernelsize: int = 16) -> np.ndarray:
    h_activitymap = img.shape[1] // kernelsize
    w_activitymap = img.shape[2] // kernelsize
    activitymap = []
    for frame in img:
        activitymap.append(
            transform_frame(frame, kernelsize, h_activitymap, w_activitymap)
        )
    return np.array(activitymap)


def get_frames_position(
    img: np.ndarray, min_z_score: float, kernelsize: int = 16
) -> list[list[int]]:
    frames_w_pos = []
    activitymap = compute_activitymap(img, kernelsize)
    above_z = np.argwhere(activitymap > min_z_score)
    for event in above_z:
        frames_w_pos.append([event[0], event[1] * kernelsize, event[2] * kernelsize])
    return frames_w_pos
