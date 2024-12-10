from neuroimage_denoiser.model.unet import UNet
import neuroimage_denoiser.utils.normalization as normalization
from neuroimage_denoiser.utils.write_file import write_file
from neuroimage_denoiser.utils.open_file import open_file
from neuroimage_denoiser.utils.convert import float_to_uint
import torch
import numpy as np
import os


class ModelWrapper:
    """
    A wrapper class for a U-Net model used for denoising 3D image sequences.

    This class provides methods to load pre-trained weights, normalize input images,
    perform denoising on image sequences, and write the denoised output to a file.

    Attributes:
        weights (str): Path to the pre-trained weights.
        batch_size (int): Number of frames to process in each batch.
        cpu (bool): Flag to force CPU usage, even if a GPU is available.
        device (torch.device): Device to use for computations (GPU or CPU).
        model (UNet): The U-Net model instance.
        denoised_img (np.ndarray): The denoised image sequence.
        img (np.ndarray): The input image sequence.
        img_height (int): Height of the input image sequence.
        img_width (int): Width of the input image sequence.
        img_mean (np.ndarray): Mean of the input image sequence along the z-axis.
        img_std (np.ndarray): Standard deviation of the input image sequence along the z-axis.
    """

    def __init__(
        self, weights: str, batch_size: int, cpu: bool, gpu_num: str, num_frames: int
    ) -> None:
        """
        Initialize the ModelWrapper instance.

        Args:
            weights (str): Path to the pre-trained weights file. Must be a valid file path.
            batch_size (int): Number of frame sequences to process in parallel. Must be positive.
            cpu (bool): Flag to force CPU usage, even if a GPU is available.
            gpu_num (str): GPU device number to use if available (e.g. "0", "1"). Ignored if cpu=True.
            num_frames (int): Number of frames per temporal window for processing. Must be positive.

        Raises:
            ValueError: If weights file does not exist, batch_size <= 0, or num_frames <= 0
        """
        if not os.path.exists(weights):
            raise ValueError(f"Weights file not found: {weights}")
        if batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {batch_size}")
        if num_frames <= 0:
            raise ValueError(f"Number of frames must be positive, got {num_frames}")
        # initalize model
        self.batch_size = batch_size
        # check for GPU, use CPU otherwise
        self.device = torch.device(
            f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu"
        )
        # if flag cpu is set, use cpu regardless of available GPU
        if cpu:
            self.device = "cpu"
        self.model = UNet(1)
        self.load_weights(weights)
        self.model.to(self.device)
        self.num_frames = num_frames
        # initalize image
        self.img_tensor = None
        self.img_num_frames = None
        self.img_mean = None
        self.img_std = None

    def load_weights(self, weights: str) -> None:
        """
        Load pre-trained weights into the U-Net model.

        Args:
            weights (str): Path to the pre-trained weights file.
        """
        self.model.load_state_dict(
            torch.load(weights, map_location=torch.device(self.device))
        )
        self.model.eval()

    def normalize_img(self, img: np.ndarray) -> None:
        """
        Normalize image using z-score normalization.

        Args:
            img: Input image of shape [frames, height, width]
        Returns:
            Normalized image of same shape
        Raises:
            ValueError: If input image is empty or has wrong dimensions
        """
        self.img_std = np.std(img, axis=0)
        self.img_mean = np.mean(img, axis=0)
        # normalization
        img: torch.tensor = normalization.z_norm(img, self.img_mean, self.img_std)
        return img

    def get_prediction_frames(self, from_frame: int) -> torch.Tensor:
        tmp = []
        to_frame = from_frame + self.batch_size * self.num_frames
        for frame_idx in range(from_frame, to_frame, self.num_frames):
            if frame_idx >= self.img_tensor.shape[0]:
                break
            elif self.img_tensor.shape[0] < frame_idx + self.num_frames:
                # if not enough frames, use mirror padding
                temporal_window = self.img_tensor[frame_idx:]
                padded_frames = self.num_frames - temporal_window.shape[0]
                temporal_window = torch.nn.functional.pad(
                    temporal_window,
                    (0, padded_frames, 0, 0, 0, 0),
                    mode="reflect",
                )
            else:
                temporal_window = self.img_tensor[
                    frame_idx : frame_idx + self.num_frames
                ]
            temporal_window = temporal_window.reshape(
                1, self.num_frames, self.img_tensor.shape[1], self.img_tensor.shape[2]
            )
            tmp.append(temporal_window)
        X = torch.empty(
            len(tmp),
            1,
            self.num_frames,
            self.img_tensor.shape[1],
            self.img_tensor.shape[2],
        )
        for i, x in enumerate(tmp):
            X[i] = x
        return X

    def denoise_img(self, img_path: str) -> None:
        """
        Denoise an image sequence using the U-Net model.

        This method loads and normalizes the input image sequence, performs
        inference using the U-Net model, and stores the denoised image sequence
        in the `denoised_img` attribute.

        Args:
            img_path (str): Path to the image sequence file.
        """
        img: np.ndarray = open_file(img_path)
        self.img_num_frames, self.img_height, self.img_width = img.shape
        img: np.ndarray = self.normalize_img(img)
        self.img_tensor: torch.tensor = torch.tensor(img, dtype=torch.float)
        # initalize bigger to account for pottential padding frames, these will
        # be removed at the end of denoise_img()
        denoised_image_sequence = torch.zeros(
            (
                self.img_tensor.shape[0] + self.num_frames,
                self.img_height,
                self.img_width,
            ),
            device="cpu",
            dtype=torch.float32,
        )
        for from_frame in range(
            0, self.img_tensor.shape[0], self.batch_size * self.num_frames
        ):
            to_frame = min(
                from_frame + self.batch_size * self.num_frames, self.img_tensor.shape[0]
            )
            X = self.get_prediction_frames(from_frame).to(self.device)
            with torch.no_grad():
                y_pred = self.model(X.type(torch.float))
            batch_predictions = y_pred.squeeze(1)
            denoised_image_sequence[from_frame:to_frame] = batch_predictions.cpu()
        denoised_image_sequence = normalization.reverse_z_norm(
            np.array(denoised_image_sequence), self.img_mean, self.img_std
        )
        # tiff format is based on uint16 -> cast
        self.denoised_img = float_to_uint(denoised_image_sequence)
        # remove potentially padded frames
        self.denoised_img = self.denoised_img[: self.img_num_frames, :, :]

    def write_denoised_img(self, outpath: str) -> None:
        """
        Write the denoised image sequence to a file.

        This method writes the denoised image sequence stored in the `denoised_img`
        attribute to the specified output file path.

        Args:
            outpath (str): Path to the output file where the denoised image sequence
                           should be written.

        Raises:
            AssertionError: If the `denoised_img` attribute is empty, indicating that
                            the image sequence has not been denoised yet.
        """
        if not hasattr(self, "denoised_img") or self.denoised_img.size == 0:
            raise ValueError("No denoised image available. Run denoise_img first.")
        try:
            write_file(self.denoised_img, outpath)
        except Exception as e:
            raise RuntimeError(f"Failed to write denoised image to {outpath}: {str(e)}")
