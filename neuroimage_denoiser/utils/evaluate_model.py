from neuroimage_denoiser.model.denoise import inference
from neuroimage_denoiser.utils.open_file import open_file

import numpy as np
from skimage.draw import disk
import roifile
import torch
import os
from scipy.signal import find_peaks


def compute_threshold(
    vals: np.ndarray,
    threshold_mult: float,
    threshold_start: int,
    threshold_stop: int,
) -> float:
    """
    Function that computes the threshold above which a peak is considered spike
    based on the formular threshold = mean + std * mult
    """
    std_ = np.std(vals[threshold_start:threshold_stop])
    mean_ = np.mean(vals[threshold_start:threshold_stop])
    return mean_ + threshold_mult * std_


def peak_detection_scipy(
    intenstiy: np.ndarray,
    threshold: float,
    stim_frames: list[int],
    patience: int,
) -> list[int]:
    """
    Peak detection using scipy.
    """
    peaks = []
    for frame in stim_frames:
        tmp_peaks, _ = find_peaks(
            intenstiy[frame - 1 : frame + patience + 1], height=threshold
        )
        peaks += [peak + frame - 1 for peak in tmp_peaks]
    return peaks


def extract_roi_trace(img: np.ndarray, roi_path: str) -> np.ndarray:
    roi = roifile.roiread(roi_path)
    # roi to circular mask
    center_x = (roi.right - roi.left) // 2 + roi.left
    center_y = (roi.bottom - roi.top) // 2 + roi.top
    radius = (roi.bottom - roi.top) // 2
    mask = disk((center_y, center_x), radius)
    return np.mean(img[:, mask[0], mask[1]], axis=1)


def raw_evaluate(
    img_path: str,
    roi_dir: str,
    stimulation_frames: list[int],
    response_patience: int,
) -> dict:
    img = open_file(img_path)
    stimulation_frames = sorted(stimulation_frames)
    possible_response_frames = []
    for stim in stimulation_frames:
        for pf in range(1, response_patience + 1):
            possible_response_frames.append(stim + pf)
    noise_stds = []
    result = {}
    for roi in os.listdir(roi_dir):
        if not roi.endswith(".roi"):
            continue
        roi_name = str(roi.split(".roi")[0])
        result[roi_name] = {}
        mean_trace = extract_roi_trace(img, os.path.join(roi_dir, roi))
        threshold = compute_threshold(mean_trace, 4, 0, stimulation_frames[0])
        peak_frames = peak_detection_scipy(
            mean_trace, threshold, stimulation_frames, response_patience
        )
        result[roi_name]["peak_frames"] = [int(frame) for frame in peak_frames]
        result[roi_name]["peak_intensities"] = mean_trace[peak_frames].tolist()
        # result[roi_name]["peak_intensities"] = mean_trace[result[roi_name]["peak_frames"]].tolist()
        noise_stds.append(
            np.std(
                [
                    val
                    for i, val in enumerate(mean_trace)
                    if i not in possible_response_frames
                ]
            )
        )
    result["noise_stds"] = float(np.mean(noise_stds))
    return result


def evaluate(
    modelpath: str,
    tmppath: str,
    img_path: str,
    batch_size: int,
    roi_dir: str,
    stimulation_frames: list[int],
    response_patience: int,
    result_raw: dict,
) -> dict:
    use_cpu = not torch.cuda.is_available()
    stimulation_frames = sorted(stimulation_frames)
    # denoise image
    inference(img_path, modelpath, False, tmppath, batch_size, use_cpu, False)
    img_name = ".".join(os.path.basename(img_path).split(".")[:-1])
    fileending = os.path.basename(img_path).split(".")[-1]
    img = open_file(os.path.join(tmppath, f"{img_name}_denoised.{fileending}"))
    # evaluate
    result = {}
    possible_response_frames = []
    for stim in stimulation_frames:
        for pf in range(1, response_patience + 1):
            possible_response_frames.append(stim + pf)
    noise_stds = []
    for roi in os.listdir(roi_dir):
        if not roi.endswith(".roi"):
            continue
        roi_name = str(roi.split(".roi")[0])
        result[roi_name] = {}
        mean_trace = extract_roi_trace(img, os.path.join(roi_dir, roi))
        threshold = compute_threshold(mean_trace, 4, 0, stimulation_frames[0])
        peak_frames = peak_detection_scipy(
            mean_trace, threshold, stimulation_frames, response_patience
        )
        result[roi_name]["peak_frames"] = [int(frame) for frame in peak_frames]
        result[roi_name]["peak_intensities"] = mean_trace[
            result[roi_name]["peak_frames"]
        ].tolist()
        result[roi_name]["peak_intensities_match_raw_events"] = mean_trace[
            result_raw[roi_name]["peak_frames"]
        ].tolist()
        noise_stds.append(
            np.std(
                [
                    val
                    for i, val in enumerate(mean_trace)
                    if i not in possible_response_frames
                ]
            )
        )
    result["noise_stds"] = float(np.mean(noise_stds))
    os.remove(os.path.join(tmppath, f"{img_name}_denoised.{fileending}"))
    return result
