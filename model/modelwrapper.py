from model.unet import UNet
import utils.normalization as normalization
from utils.open_file import open_file
import torch
import numpy as np
from tqdm import tqdm


class ModelWrapper:
    """
    Wrapper class for a U-Net model used for image denoising.

    Parameters:
    - weights (str): Path to the pre-trained weights file.
    - n_pre (int): Number of frames to use before the target frame.
    - n_post (int): Number of frames to use after the target frame.
    """

    def __init__(self, weights: str, n_pre: int, n_post: int) -> None:
        """
        Initialize the ModelWrapper.

        Initializes the U-Net model, loads pre-trained weights, and sets up device (GPU or CPU).

        Parameters:
        - weights (str): Path to the pre-trained weights file.
        - n_pre (int): Number of frames to use before the target frame.
        - n_post (int): Number of frames to use after the target frame.
        """
        # initalize model
        self.n_pre = n_pre
        self.n_post = n_post
        # check for GPU, use CPU otherwise
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet(self.n_pre + self.n_post)
        self.load_weights(weights)
        self.model.to(self.device)
        # initalize image
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
        self.model.load_state_dict(torch.load(weights))
        self.model.eval()

    def load_img(self, img_path: str) -> None:
        """
        Load an image from the specified path, perform normalization, and store information about the image.

        Parameters:
        - img_path (str): Path to the image file.
        """
        self.img = open_file(img_path)
        _, self.img_height, self.img_width = self.img.shape
        # normalization
        self.img_mean: np.ndarray = np.mean(self.img, axis=0)
        self.img_std: np.ndarray = np.std(self.img, axis=0)
        self.img: np.ndarray = normalization.z_norm(
            self.img, self.img_mean, self.img_std
        )

    def get_prediction_frames(self, target: int) -> torch.Tensor:
        """
        Extract frames around the target frame for making predictions.

        Parameters:
        - target (int): Index of the target frame.

        Returns:
        - torch.Tensor: Input tensor for the U-Net model.
        """
        # extract frames
        X = self.img[target - self.n_pre - 1 : target + self.n_post]  # ignore: warning
        # remove target frame
        X = np.delete(X, self.n_pre, axis=0)
        # reshape to batch size 1
        X = X.reshape(1, self.n_pre + self.n_post, self.img_height, self.img_width)
        return torch.tensor(X, dtype=torch.float)

    def denoise_img(self, img_path: str) -> np.ndarray:
        """
        Denoise an image sequence using the U-Net model.

        Parameters:
        - img_path (str): Path to the image sequence file.

        Returns:
        - np.ndarray: Denoised image sequence.
        """
        denoised_image_sequence = []
        self.load_img(img_path)
        for target in tqdm(
            range(self.n_pre + 1, len(self.img) - self.n_post), desc="denoise"
        ):
            X = self.get_prediction_frames(target).to(self.device)
            y_pred = np.array(self.model(X).detach().to("cpu")).reshape(
                self.img_height, self.img_width
            )
            denoised_image_sequence.append(y_pred)
        y_pred_grey_vals = normalization.reverse_z_norm(
            np.array(denoised_image_sequence), self.img_mean, self.img_std
        )
        return y_pred_grey_vals
