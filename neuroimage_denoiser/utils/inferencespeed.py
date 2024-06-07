from neuroimage_denoiser.model.modelwrapper import ModelWrapper
from neuroimage_denoiser.utils.open_file import open_file
import neuroimage_denoiser.utils.normalization as normalization
import numpy as np
import os
import time
import json
from alive_progress import alive_bar


def crop_img(img: np.ndarray, cropsize: int, num_frames: int) -> np.ndarray:
    """
    Crop an image sequence to random square regions across multiple frames.

    Args:
        img (np.ndarray): Input image sequence tensor of shape (frames, height, width).
        cropsize (int): Size of the square crop region.
        num_frames (int): Number of frames to crop from the input image sequence.

    Returns:
        np.ndarray: Cropped image sequence tensor of shape (num_frames, cropsize, cropsize).

    Raises:
        ValueError: If the crop size is larger than the input image dimensions or if the number of frames is greater than the number of frames in the input image sequence.
    """
    if cropsize > img.shape[1] or cropsize > img.shape[2]:
        raise ValueError("Crop size cannot be larger than the input image dimensions.")

    if num_frames > img.shape[0]:
        raise ValueError(
            "Number of frames cannot be greater than the number of frames in the input image sequence."
        )
    n_start = np.random.randint(0, img.shape[0] - num_frames)
    y_start = np.random.randint(0, img.shape[1] - cropsize)
    y_end = y_start + cropsize
    x_start = np.random.randint(0, img.shape[2] - cropsize)
    x_end = x_start + cropsize
    return img[n_start : n_start + num_frames, y_start:y_end, x_start:x_end]


def eval_inferencespeed(
    modelpath: str,
    folderpath: str,
    cropsizes: list[int],
    num_frames: int,
    cpu: bool,
    outpath: str,
) -> None:
    """
    Evaluate the inference speed of a U-Net model on cropped image sequences.

    Args:
        modelpath (str): Path to the pre-trained model weights.
        folderpath (str): Path to the folder containing input image sequences.
        cropsizes (List[int]): List of crop sizes to evaluate.
        num_frames (int): Number of frames to crop from each input image sequence.
        cpu (bool): Flag to force CPU usage, even if a GPU is available.
        outpath (str): Path to the output directory for saving the inference speed results.

    Raises:
        ValueError: If the specified number of frames exceeds the number of frames in an input image sequence.
        FileNotFoundError: If the model weights file or the input folder does not exist.
        OSError: If there is an issue creating the output directory.

    Returns:
        None
    """
    if not os.path.exists(modelpath):
        raise FileNotFoundError(f"Model weights file not found: {modelpath}")

    if not os.path.exists(folderpath):
        raise FileNotFoundError(f"Input folder not found: {folderpath}")

    if not os.path.exists(outpath):
        try:
            os.makedirs(outpath)
        except OSError as e:
            raise OSError(f"Error creating output directory: {outpath}") from e

    fileendings = [".tif", ".tiff", ".stk", ".nd2"]
    runtimes = {}
    model = ModelWrapper(modelpath, 1, cpu)
    model.load_weights(modelpath)
    with alive_bar(len(cropsizes)) as bar:
        for cropsize in sorted(cropsizes):
            runtimes[cropsize] = []
            for filename in os.listdir(folderpath):
                filepath = os.path.join(folderpath, filename)
                if not any(
                    [filepath.endswith(fileending) for fileending in fileendings]
                ):
                    continue
                img = open_file(filepath)
                if num_frames > img.shape[0]:
                    raise ValueError(
                        f"Out of bound, num_frames exceeds frames in {filepath} ({num_frames}>{img.shape[0]})"
                    )
                img_croped = crop_img(img, cropsize, num_frames)
                start = time.time()
                model.img = img_croped
                model.img_height = cropsize
                model.img_width = cropsize
                model.normalize_img()
                denoised_image_sequence = model.inference()
                denoised_image_sequence = normalization.reverse_z_norm(
                    denoised_image_sequence, model.img_mean, model.img_std
                )
                runtimes[cropsize].append(time.time() - start)
            bar()
    outfile = os.path.join(outpath, "inferencespeed.json")
    with open(outfile, "w") as f:
        json.dump(runtimes, f)
