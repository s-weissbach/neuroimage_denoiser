from neuroimage_denoiser.model.unet import UNet
import neuroimage_denoiser.utils.normalization as normalization
from neuroimage_denoiser.utils.write_file import write_file
from neuroimage_denoiser.utils.open_file import open_file
from neuroimage_denoiser.utils.convert import float_to_uint
import torch
import numpy as np


class ModelWrapper:
    """
    Wrapper class for a U-Net model used for image denoising.
    """

    def __init__(self, n_frames: int, weights: str, cpu: bool) -> None:
        """
        Initialize the ModelWrapper class for image denoising using a U-Net model.

        Parameters:
        - n_frames (int): Number of frames to consider for denoising.
        - weights (str): Path to the pre-trained weights file for the U-Net model.
        - cpu (bool): Flag to indicate whether to use CPU for inference, overriding GPU availability.

        Attributes:
        - n_frames (int): Number of frames used for denoising.
        - device (torch.device): Device used for inference (CPU or GPU).
        - model (UNet): U-Net model for image denoising.
        - denoised_img (np.ndarray): Denoised image sequence.
        - img (np.ndarray): Loaded image sequence.
        - img_height (int): Height of the image frames.
        - img_width (int): Width of the image frames.
        - img_mean (np.ndarray): Mean pixel values of the image frames.
        - img_std (np.ndarray): Standard deviation of pixel values of the image frames.
        """
        self.n_frames = n_frames
        # check for GPU, use CPU otherwise
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # if flag cpu is set, use cpu regardless of available GPU
        if cpu:
            self.device = "cpu"
        self.model = UNet(n_frames)
        self.load_weights(weights)
        self.model.to(self.device)
        # initalize image
        self.denoised_img = np.empty((0, 0, 0))
        self.img = np.empty((0, 0, 0))
        self.img_height = -1
        self.img_width = -1
        self.img_mean = np.empty((0, 0))
        self.img_std = np.empty((0, 0))

    def load_weights(self, weights: str) -> None:
        """
        Load pre-trained weights into the U-Net model.

        Parameters:
        - weights (str): Path to the pre-trained weights file.
        """
        if self.device == "cpu":
            self.model.load_state_dict(
                torch.load(weights, map_location=torch.device("cpu"))
            )
        else:
            self.model.load_state_dict(torch.load(weights))
        self.model.eval()

    def load_and_normalize_img(self, img_path: str) -> None:
        """
        Load an image from the specified path, perform normalization, and store information about the image.

        Parameters:
        - img_path (str): Path to the image file.
        """
        self.img: np.ndarray = open_file(img_path)
        _, self.img_height, self.img_width = self.img.shape
        # compute mean and std along z-axis
        self.img_mean: np.ndarray = np.mean(self.img, axis=0)
        self.img_std: np.ndarray = np.std(self.img, axis=0)
        # normalization
        self.img: np.ndarray = normalization.z_norm(
            self.img, self.img_mean, self.img_std
        )

    def get_prediction_frames(self, target_frame: int) -> torch.Tensor:
        """
        Extract frames around the target frame for making predictions.

        Parameters:
        - target (int): Index of the target frame.

        Returns:
        - torch.Tensor: Input tensor for the U-Net model.
        """

        # extract frames
        from_frame = target_frame - (self.n_frames // 2)
        to_frame = target_frame + (self.n_frames // 2 + 1)
        X = self.img[
            from_frame:to_frame,
            :,
            :,
        ]
        X = X.reshape(self.n_frames + 1, self.img_height, self.img_width)
        X = np.delete(X, self.n_frames // 2, axis=0)
        # reshape to batch size 1
        X = X.reshape(
            1,
            self.n_frames,
            self.img_height,
            self.img_width,
        )
        return torch.tensor(X, dtype=torch.float)

    def denoise_img(self, img_path: str) -> None:
        """
        Denoise an image sequence using the U-Net model.

        Parameters:
        - img_path (str): Path to the image sequence file.

        Returns:
        - np.ndarray: Denoised image sequence.
        """
        denoised_image_sequence = []
        self.load_and_normalize_img(img_path)
        for target_frame in range(
            self.n_frames // 2, self.img.shape[0] - self.n_frames // 2
        ):
            X = self.get_prediction_frames(target_frame).to(self.device)
            y_pred = np.array(self.model(X).detach().to("cpu"))
            for denoised_frame in y_pred:
                denoised_image_sequence.append(
                    denoised_frame.reshape(self.img_height, self.img_width)
                )
        self.denoised_img = normalization.reverse_z_norm(
            np.array(denoised_image_sequence), self.img_mean, self.img_std
        )
        # tiff format is based on uint16 -> cast
        self.denoised_img = float_to_uint(self.denoised_img)

    def write_denoised_img(self, outpath: str) -> None:
        if self.denoised_img.shape[0] == 0:
            raise AssertionError(
                f"Before writing a denoised image, first denoise image. Use <ModelWrapper>.denoise_img(<path/to/input_image>)."
            )
        write_file(self.denoised_img, outpath)
